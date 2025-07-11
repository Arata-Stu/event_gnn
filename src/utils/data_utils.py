import numpy as np
import torch
from torch_geometric.data import Data


def to_data(**kwargs):
    # convert all tracks to correct format
    for k, v in kwargs.items():
        if k.startswith("bbox"):
            kwargs[k] = torch.from_numpy(v)

    xy = np.stack([kwargs['x'], kwargs['y']], axis=-1).astype("int16")
    t = kwargs['t'].astype("int32")
    p = kwargs['p'].reshape((-1,1))

    kwargs['x'] = torch.from_numpy(p)
    kwargs['pos'] = torch.from_numpy(xy)
    kwargs['t'] = torch.from_numpy(t)

    return Data(**kwargs)

def format_data(data, normalizer=None):
    if normalizer is None:
        normalizer = torch.stack([data.width[0], data.height[0], data.time_window[0]], dim=-1)

    if hasattr(data, "image"):
        data.image = data.image.float() / 255.0

    data.pos = torch.cat([data.pos, data.t.view((-1,1))], dim=-1)
    data.t = None
    data.x = data.x.float()
    data.pos = data.pos / normalizer
    return data