import sys
sys.path.append('../')

from pathlib import Path
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from src.model.networks.net import Net
from src.data.dataset.dsec.dataset_for_graph import DSEC
from src.utils.data_utils import format_data

data_path = '/media/arata-22/AT_2TB/dataset/dsec'
data_path = Path(data_path).resolve()

split = 'train'
dataset = DSEC(
    root=data_path,
    split=split,
    transform=None,
    debug=False,
    min_bbox_diag=15,
    min_bbox_height=10,
    scale=2,
    cropped_height=430,
    only_perfect_tracks=True,
    demo=False,
    no_eval=False)

batch_size = 1  # バッチサイズを1に設定
test_loader = DataLoader(dataset, follow_batch=['bbox', "bbox0"], batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)


cfg_path = '../config/default.yaml'
# OmegaConf.create()でDictConfigに変換
cfg = OmegaConf.load(cfg_path)

# --- モデルの初期化（変更なし） ---
model = Net(cfg.model, 240, 320)
model.eval()
model.cuda()
# model.cache_luts(radius=cfg.model.ev_graph.radius, height=dataset.height, width=dataset.width)

for i, data in enumerate(test_loader):
    if i >= 100:  # 最初の1つのバッチのみをテスト
        break
    data = data.cuda(non_blocking=True)
    data = format_data(data)

    output = model(data, reset=True)
    print(f"Output for batch {i}: {output}")
