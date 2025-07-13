import torch
from torch import nn
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.model.backbone.ev_gnn import EVGNNBackbone

class Net(nn.Module):
    def __init__(self, cfg: DictConfig, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width
        self.backbone = EVGNNBackbone(cfg, height, width)

    def cache_luts(self, width, height, radius):
        M = 2 * float(int(radius * width + 2) / width)
        r = int(radius * width+1)
        self.backbone.conv_block1.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=r)
        self.backbone.conv_block1.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=r)

        # rx, ry, M = voxel_size_to_params(self.backbone.pool1, height, width)
        # self.backbone.layer2.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        # self.backbone.layer2.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        # rx, ry, M = voxel_size_to_params(self.backbone.pool2, height, width)
        # self.backbone.layer3.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        # self.backbone.layer3.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        # rx, ry, M = voxel_size_to_params(self.backbone.pool3, height, width)
        # self.backbone.layer4.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        # self.backbone.layer4.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        # self.head.stem1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        # self.head.cls_conv1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        # self.head.reg_conv1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        # self.head.cls_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        # self.head.reg_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        # self.head.obj_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        # rx, ry, M = voxel_size_to_params(self.backbone.pool4, height, width)
        # self.backbone.layer5.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        # self.backbone.layer5.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        # if self.head.num_scales > 1:
        #     self.head.stem2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        #     self.head.cls_conv2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        #     self.head.reg_conv2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        #     self.head.cls_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        #     self.head.reg_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        #     self.head.obj_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        
    def forward(self, data: Data, reset=True):
        data = self.backbone(data, reset=reset)
        # data = self.head(data)
        return data