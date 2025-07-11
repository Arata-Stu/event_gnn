import sys
sys.path.append('../')

from torch.utils.data import DataLoader
from src.data.dataset.dsec.dataset_for_graph import DSEC

data_path = '../data/dsec'
split = 'train'
dataset = DSEC(
    root=data_path,
    split=split,
    transform=None,
    debug=False,
    min_bbox_diag=15,
    min_bbox_height=10,
    scale=2.0,
    cropped_height=430,
    only_perfect_tracks=True,
    demo=False,
    no_eval=False)

batch_size = 1  # バッチサイズを1に設定
test_loader = DataLoader(dataset, follow_batch=['bbox', "bbox0"], batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
