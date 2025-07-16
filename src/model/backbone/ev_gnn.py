import torch 
from torch import nn
from omegaconf import DictConfig
from torch_geometric.data import Data
import torch_geometric.transforms as T

from ..layers.ev_tgn import EV_TGN
from ..layers.pooling import Pooling
from ..layers.components import Cartesian
from ..layers.conv import Layer
from ..utils import compute_pooling_at_each_layer, shallow_copy
from src.utils.timers import CudaTimer as Timer

class EVGNNBackbone(nn.Module):
    def __init__(self, cfg: DictConfig, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width
        
        ## 出力される特徴マップのスケール種類
        self.num_scales = cfg.num_scales

        channels = [1,
                    int(cfg.base_width*32),
                    int(cfg.after_pool_width*64),
                    int(cfg.net_stem_width*128),
                    int(cfg.net_stem_width*128),
                    int(cfg.net_stem_width*128)]
        
        input_channels = channels[:-1]
        output_channels = channels[1:]

        self.events_to_graph = EV_TGN(cfg.ev_graph)

        # 構造化されたconfigを想定 (例: cfg.pool.pooling_dim_at_output)
        poolings = compute_pooling_at_each_layer(cfg.pool.pooling_dim_at_output, num_layers=4)
        max_vals_for_cartesian = 2*poolings[:,:2].max(-1).values
        self.strides = torch.ceil(poolings[-2:,1] * height).numpy().astype("int32").tolist()
        self.strides = self.strides[-self.num_scales:]

        effective_radius = 2*float(int(cfg.ev_graph.radius * width + 2) / width)
        self.edge_attrs = Cartesian(norm=True, cat=False, max_value=effective_radius)

        # Layerには畳み込み層関連のconfigのみを渡すのが良い設計です
        self.conv_block1 = Layer(2+input_channels[0], output_channels[0], cfg=cfg.conv)

        cart1 = T.Cartesian(norm=True, cat=False, max_value=2*effective_radius)
        self.pool1 = Pooling(poolings[0], width=width, height=height, batch_size=cfg.batch_size,
                             transform=cart1, aggr=cfg.pool.pooling_aggr, keep_temporal_ordering=cfg.pool.keep_temporal_ordering)
        
        self.layer2 = Layer(input_channels[1]+2, output_channels[1], cfg=cfg.conv)

        cart2 = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[1])
        self.pool2 = Pooling(poolings[1], width=width, height=height, batch_size=cfg.batch_size,
                             transform=cart2, aggr=cfg.pool.pooling_aggr, keep_temporal_ordering=cfg.pool.keep_temporal_ordering)

        self.layer3 = Layer(input_channels[2]+2, output_channels[2],  cfg=cfg.conv)

        cart3 = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[2])
        self.pool3 = Pooling(poolings[2], width=width, height=height, batch_size=cfg.batch_size,
                             transform=cart3, aggr=cfg.pool.pooling_aggr, keep_temporal_ordering=cfg.pool.keep_temporal_ordering)

        self.layer4 = Layer(input_channels[3]+2, output_channels[3],  cfg=cfg.conv)

        cart4 = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[3])
        self.pool4 = Pooling(poolings[3], width=width, height=height, batch_size=cfg.batch_size,
                             transform=cart4, aggr='mean', keep_temporal_ordering=cfg.pool.keep_temporal_ordering)

        self.layer5 = Layer(input_channels[4]+2, output_channels[4],  cfg=cfg.conv)

        self.cache = []

    def forward(self, data: Data, reset=True):
        if hasattr(data, 'reset'):
            reset = data.reset

        with Timer(data.x.device, 'EVGNNBackbone.forward.events_to_graph'):
            data = self.events_to_graph(data, reset=reset)

        with Timer(data.x.device, 'EVGNNBackbone.forward.cartesian'):
            data = self.edge_attrs(data)
            data.edge_attr = torch.clamp(data.edge_attr, min=0, max=1)
            
        # --- Stage 1 & 2 ---
        with Timer(data.x.device, 'EVGNNBackbone.forward.conv1_pool1_layer2'):
            rel_delta = data.pos[:, :2]
            data.x = torch.cat((data.x, rel_delta), dim=1)
            data = self.conv_block1(data)
            
            data = self.pool1(data)
            
            rel_delta = data.pos[:,:2]
            data.x = torch.cat((data.x, rel_delta), dim=1)
            data = self.layer2(data)

        # --- Stage 3 ---
        with Timer(data.x.device, 'EVGNNBackbone.forward.pool2_layer3'):
            data = self.pool2(data)
            rel_delta = data.pos[:,:2]
            data.x = torch.cat((data.x, rel_delta), dim=1)
            data = self.layer3(data)

        # --- Stage 4 ---
        with Timer(data.x.device, 'EVGNNBackbone.forward.pool3_layer4'):
            data = self.pool3(data)
            rel_delta = data.pos[:,:2]
            data.x = torch.cat((data.x, rel_delta), dim=1)
            data = self.layer4(data)

        # ★ 修正点: out3 を layer4 の直後、pool4 の前で保存します
        out3 = shallow_copy(data)
        out3.pooling = self.pool3.voxel_size[:3]

        # --- Stage 5 ---
        with Timer(data.x.device, 'EVGNNBackbone.forward.pool4_layer5'):
            data = self.pool4(data)
            rel_delta = data.pos[:,:2]
            data.x = torch.cat((data.x, rel_delta), dim=1)
            data = self.layer5(data)    

        out4 = data
        out4.pooling = self.pool4.voxel_size[:3]

        output = [out3, out4]

        return output[-self.num_scales:]