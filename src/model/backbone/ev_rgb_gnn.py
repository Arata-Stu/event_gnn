import torch 
from torch import nn
from omegaconf import DictConfig
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

from ..layers.ev_tgn import EV_TGN
from ..layers.pooling import Pooling
from ..layers.components import Cartesian
from ..layers.conv import Layer
from ..networks.net_img import HookModule
from ..utils import compute_pooling_at_each_layer, shallow_copy
from src.utils.timers import CudaTimer as Timer

weight = {
    "resnet18": ResNet18_Weights.DEFAULT,
    "resnet34": ResNet34_Weights.DEFAULT,
    "resnet50": ResNet50_Weights.DEFAULT
}

def sampling_skip(data, image_feat):
    image_feat_at_nodes = sample_features(data, image_feat)
    return torch.cat((data.x, image_feat_at_nodes), dim=1)


def sample_features(data, image_feat, image_sample_mode="bilinear"):
    if data.batch is None or len(data.batch) != len(data.pos):
        data.batch = torch.zeros(len(data.pos), dtype=torch.long, device=data.x.device)
    return _sample_features(data.pos[:,0] * data.width[0],
                            data.pos[:,1] * data.height[0],
                            data.batch.float(), image_feat,
                            data.width[0],
                            data.height[0],
                            image_feat.shape[0],
                            image_sample_mode)

def _sample_features(x, y, b, image_feat, width, height, batch_size, image_sample_mode):
    x = 2 * x / (width - 1) - 1
    y = 2 * y / (height - 1) - 1

    batch_size = batch_size if batch_size > 1 else 2
    b = 2 * b / (batch_size - 1) - 1

    grid = torch.stack((x, y, b), dim=-1).view(1, 1, 1,-1, 3) # N x D_out x H_out x W_out x 3 (N=1, D_out=1, H_out=1)
    image_feat = image_feat.permute(1,0,2,3).unsqueeze(0) # N x C x D x H x W (N=1)

    image_feat_sampled = torch.nn.functional.grid_sample(image_feat,
                                                         grid=grid,
                                                         mode=image_sample_mode,
                                                         align_corners=True) # N x C x H_out x W_out (H_out=1, N=1)

    image_feat_sampled = image_feat_sampled.view(image_feat.shape[1], -1).t()

    return image_feat_sampled


class EVRGBGNNBackbone(nn.Module):
    def __init__(self, model_cfg: DictConfig, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width

        self.num_classes = 2
        
        ## 出力される特徴マップのスケール種類
        self.num_scales = model_cfg.num_scales
        self.use_image = model_cfg.use_image

        channels = [1,
                    int(model_cfg.base_width*32),
                    int(model_cfg.after_pool_width*64),
                    int(model_cfg.net_stem_width*128),
                    int(model_cfg.net_stem_width*128),
                    int(model_cfg.net_stem_width*128)]

        self.out_channels_cnn = []
        if self.use_image:
            img_net = eval(model_cfg.img_net)
            weights = weight[model_cfg.img_net] if model_cfg.img_net in weight else None
            self.out_channels_cnn = [256, 256]
            self.net = HookModule(img_net(weights=weights),
                                  input_channels=3,
                                  height=height, width=width,
                                  feature_layers=["conv1", "layer1", "layer2", "layer3", "layer4"],
                                  output_layers=["layer3", "layer4"],
                                  feature_channels=channels[1:],
                                  output_channels=self.out_channels_cnn)

        
        input_channels = channels[:-1]
        if self.use_image:
            input_channels = [input_channels[i] + self.net.feature_channels[i] for i in range(len(input_channels))]

        output_channels = channels[1:]

        self.events_to_graph = EV_TGN(model_cfg.ev_graph)

        poolings = compute_pooling_at_each_layer(model_cfg.pool.pooling_dim_at_output, num_layers=4)
        max_vals_for_cartesian = 2*poolings[:,:2].max(-1).values
        self.strides = torch.ceil(poolings[-2:,1] * height).numpy().astype("int32").tolist()
        self.strides = self.strides[-self.num_scales:]

        effective_radius = 2*float(int(model_cfg.ev_graph.radius * width + 2) / width)
        self.edge_attrs = Cartesian(norm=True, cat=False, max_value=effective_radius)

        self.conv_block1 = Layer(2+input_channels[0], output_channels[0], cfg=model_cfg.conv)

        cart1 = T.Cartesian(norm=True, cat=False, max_value=2*effective_radius)
        self.pool1 = Pooling(poolings[0], width=width, height=height, batch_size=model_cfg.batch_size,
                             transform=cart1, aggr=model_cfg.pool.pooling_aggr, keep_temporal_ordering=model_cfg.pool.keep_temporal_ordering)

        self.layer2 = Layer(input_channels[1]+2, output_channels[1], cfg=model_cfg.conv)

        cart2 = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[1])
        self.pool2 = Pooling(poolings[1], width=width, height=height, batch_size=model_cfg.batch_size,
                             transform=cart2, aggr=model_cfg.pool.pooling_aggr, keep_temporal_ordering=model_cfg.pool.keep_temporal_ordering)

        self.layer3 = Layer(input_channels[2]+2, output_channels[2], cfg=model_cfg.conv)

        cart3 = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[2])
        self.pool3 = Pooling(poolings[2], width=width, height=height, batch_size=model_cfg.batch_size,
                             transform=cart3, aggr=model_cfg.pool.pooling_aggr, keep_temporal_ordering=model_cfg.pool.keep_temporal_ordering)

        self.layer4 = Layer(input_channels[3]+2, output_channels[3],  cfg=model_cfg.conv)

        cart4 = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[3])
        self.pool4 = Pooling(poolings[3], width=width, height=height, batch_size=model_cfg.batch_size,
                             transform=cart4, aggr='mean', keep_temporal_ordering=model_cfg.pool.keep_temporal_ordering)

        self.layer5 = Layer(input_channels[4]+2, output_channels[4],  cfg=model_cfg.conv)

        self.cache = []

    def forward(self, data: Data, reset=True):

        if self.use_image:
            with Timer(data.x.device, 'EVGNNBackbone.forward.net'):
                image_feat, image_outputs = self.net(data.image)
            
        if hasattr(data, 'reset'):
            reset = data.reset

        with Timer(data.x.device, 'EVGNNBackbone.forward.events_to_graph'):
            data = self.events_to_graph(data, reset=reset)

        if self.use_image:
            data.x = sampling_skip(data, image_feat[0].detach())
            data.skipped = True
            data.num_image_channels = image_feat[0].shape[1]

        with Timer(data.x.device, 'EVGNNBackbone.forward.cartesian'):
            data = self.edge_attrs(data)
            data.edge_attr = torch.clamp(data.edge_attr, min=0, max=1)
            
        with Timer(data.x.device, 'EVGNNBackbone.forward.conv_block1'):
            rel_delta = data.pos[:, :2]
            data.x = torch.cat((data.x, rel_delta), dim=1)

            data = self.conv_block1(data)

            if self.use_image:
                data.x = sampling_skip(data, image_feat[1].detach())

            data = self.pool1(data)

        if self.use_image:
            data.skipped = True
            data.num_image_channels = image_feat[1].shape[1]

                    
        with Timer(data.x.device, 'EVGNNBackbone.forward.layer2'):
            rel_delta = data.pos[:,:2]
            data.x = torch.cat((data.x, rel_delta), dim=1)

            data = self.layer2(data)

            if self.use_image:
                data.x = sampling_skip(data, image_feat[2].detach())

            data = self.pool2(data)

        if self.use_image:
            data.skipped = True
            data.num_image_channels = image_feat[2].shape[1]


        with Timer(data.x.device, 'EVGNNBackbone.forward.layer3'):
            rel_delta = data.pos[:,:2]
            data.x = torch.cat((data.x, rel_delta), dim=1)

            data = self.layer3(data)
            if self.use_image:
                data.x = sampling_skip(data, image_feat[3].detach())

            data = self.pool3(data)

        if self.use_image:
            data.skipped = True
            data.num_image_channels = image_feat[3].shape[1]


        with Timer(data.x.device, 'EVGNNBackbone.forward.layer4'):
            rel_delta = data.pos[:,:2]
            data.x = torch.cat((data.x, rel_delta), dim=1)

            data = self.layer4(data)

            if self.use_image:
                data.x = sampling_skip(data, image_feat[3].detach())

            data = self.pool4(data) 

        if self.use_image:
            data.skipped = True
            data.num_image_channels = image_feat[4].shape[1]


        out3 = shallow_copy(data)
        out3.pooling = self.pool3.voxel_size[:3]

        with Timer(data.x.device, 'EVGNNBackbone.forward.layer5'):
            rel_delta = data.pos[:,:2]
            data.x = torch.cat((data.x, rel_delta), dim=1)

            data = self.layer5(data)    

        out4 = data
        out4.pooling = self.pool4.voxel_size[:3]

        output = [out3, out4]

        if self.use_image:
            return output[-self.num_scales:], image_outputs[-self.num_scales:]

        return output[-self.num_scales:]