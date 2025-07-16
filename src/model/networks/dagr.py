import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from omegaconf import DictConfig

from ..backbone.ev_rgb_gnn import EVRGBGNNBackbone
from ..utils import init_subnetwork, voxel_size_to_params
from ..head.gnn_head import GNNHead

class DAGR(nn.Module):
    def __init__(self, cfg: DictConfig , height, width):
        super().__init__()
        self.conf_threshold = 0.001
        self.nms_threshold = 0.65

        self.height = height
        self.width = width

        self.backbone = EVRGBGNNBackbone(cfg, height=height, width=width)
        self.head = GNNHead(num_classes=self.backbone.num_classes,
                       in_channels=self.backbone.out_channels,
                       in_channels_cnn=self.backbone.out_channels_cnn,
                       strides=self.backbone.strides,
                       pretrain_cnn=cfg.pretrain_cnn,
                       args=cfg)

        
        if "img_net_checkpoint" in cfg and cfg.img_net_checkpoint is not None:
            state_dict = torch.load(cfg.img_net_checkpoint)
            init_subnetwork(self, state_dict['ema'], "backbone.net.", freeze=True)
            init_subnetwork(self, state_dict['ema'], "head.cnn_head.")

    def cache_luts(self, width, height, radius):
        M = 2 * float(int(radius * width + 2) / width)
        r = int(radius * width+1)
        self.backbone.conv_block1.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=r)
        self.backbone.conv_block1.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=r)

        rx, ry, M = voxel_size_to_params(self.backbone.pool1, height, width)
        self.backbone.layer2.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer2.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        rx, ry, M = voxel_size_to_params(self.backbone.pool2, height, width)
        self.backbone.layer3.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer3.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        rx, ry, M = voxel_size_to_params(self.backbone.pool3, height, width)
        self.backbone.layer4.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer4.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        self.head.stem1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.cls_conv1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.reg_conv1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.cls_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.reg_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.obj_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        rx, ry, M = voxel_size_to_params(self.backbone.pool4, height, width)
        self.backbone.layer5.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer5.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        if self.head.num_scales > 1:
            self.head.stem2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.cls_conv2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.reg_conv2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.cls_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.reg_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.obj_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

    def forward(self, x: Data, reset=True, targets=None):
        if not hasattr(self.head, "output_sizes"):
            self.head.output_sizes = self.backbone.get_output_sizes()

        if self.training:
            ## 損失を返す
            assert targets is not None
            # gt_target inputs need to be [l cx cy w h] in pixels
            backbone_features = self.backbone(x, reset=reset)
            outputs = self.head(backbone_features, labels=targets)
        else:
            x.reset = reset
            ## 検出結果を返す
            backbone_features = self.backbone(x, reset=reset)
            outputs = self.head(backbone_features, labels=None)

        return outputs

        # x.reset = reset

        # outputs = YOLOX.forward(self, x)

        # detections = postprocess_network_output(outputs, self.backbone.num_classes, self.conf_threshold, self.nms_threshold, filtering=filtering,
        #                                         height=self.height, width=self.width)

        # ret = [detections]

        # if return_targets and hasattr(x, 'bbox'):
        #     targets = convert_to_evaluation_format(x)
        #     ret.append(targets)

        # return ret
