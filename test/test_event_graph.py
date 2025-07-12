import sys
sys.path.append('../')

from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.model.layers.ev_tgn import EV_TGN
from src.data.dataset.dsec.dataset_for_graph import DSEC
from src.utils.data_utils import format_data

data_path = '../data/dsec'
data_path = Path(data_path).resolve()

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


cfg_dict = {
    'radius': 0.01,
    'max_neighbors': 16,
    'max_queue_size': 128
}
# OmegaConf.create()でDictConfigに変換
cfg = OmegaConf.create(cfg_dict)

# --- モデルの初期化（変更なし） ---
model = EV_TGN(cfg)

assert model.radius == cfg.radius
assert model.max_neighbors == cfg.max_neighbors
assert model.max_queue_size == cfg.max_queue_size

for i, data in enumerate(test_loader):
    if i >= 10:  # 最初の1つのバッチのみをテスト
        break
    data = data.cuda(non_blocking=True)
    data = format_data(data)

    graph = model(data)
    print(f"Graph created for batch {i}: {graph}")
    print(f"Graph nodes: {graph.num_nodes}, edges: {graph.num_edges}")  # グラフのノードとエッジ数を表示