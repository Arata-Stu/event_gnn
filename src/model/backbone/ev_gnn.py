import torch 
from torch import nn
from omegaconf import DictConfig
from torch_geometric.data import Data

from ..layers.ev_tgn import EV_TGN
from ..layers.components import Cartesian
from ..utils import compute_pooling_at_each_layer

class EVGNN(nn.Module):
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

        poolings = compute_pooling_at_each_layer(cfg.pooling_dim_at_output, num_layers=4)
        max_vals_for_cartesian = 2*poolings[:,:2].max(-1).values
        self.strides = torch.ceil(poolings[-2:,1] * height).numpy().astype("int32").tolist()
        self.strides = self.strides[-self.num_scales:]

        effective_radius = 2*float(int(cfg.ev_graph.radius * width + 2) / width)
        self.edge_attrs = Cartesian(norm=True, cat=False, max_value=effective_radius)

    def forward(self, x: Data, reset=True):
        if hasattr(data, 'reset'):
            reset = data.reset

        data = self.events_to_graph(data, reset=reset)