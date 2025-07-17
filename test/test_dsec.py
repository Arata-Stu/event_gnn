import sys
import cv2
import numpy as np 
from pathlib import Path
from torch_geometric.loader import DataLoader

from src.data.dataset.dsec.dataset_for_graph import DSEC
from src.visualization.event_viz import draw_events_on_image, map_polarity_to_channel_index

# --- データセットのセットアップ ---
data_path = Path('/media/arata-22/AT_2TB/dataset/dsec').resolve()
split = 'train'

# DSECデータセットの初期化
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
    no_eval=False
)

# DataLoaderのセットアップ
# バッチサイズを1に設定し、単一のデータインスタンスを処理
batch_size = 1
test_loader = DataLoader(
    dataset, 
    follow_batch=['bbox', "bbox0"], 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=0, 
    drop_last=True
)

# --- イベントデータの可視化ループ ---
print("イベントデータの可視化を開始します。")
print("ウィンドウが表示されたら、任意のキーを押すと次の画像へ、'q'キーを押すと終了します。")

for i, data in enumerate(test_loader):
    # 最初の1つのバッチのみを処理（必要に応じて数を増減）
    if i >= 1:
        break
    
    print(f"\n--- Batch {i+1} の処理中 ---")

    # 1. 画像データの準備
    img = data.image.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # 2. イベント座標 (x, y) と極性 (p) データの準備
    x_coords = data.pos[:, 0].cpu().numpy()
    y_coords = data.pos[:, 1].cpu().numpy()
    p_polarity = data.x.cpu().numpy().squeeze() # 極性データを1次元配列にsqueeze

    mapped_p_for_drawing = map_polarity_to_channel_index(p_polarity)


    # 3. イベントを画像に描画
    img = draw_events_on_image(
        img.copy(), # 元の画像を保持するためにコピーを渡す
        x_coords,
        y_coords,
        mapped_p_for_drawing,
        alpha=0.5
    )
    
    # image_display_bgr = cv2.cvtColor(image_with_events_rgb, cv2.COLOR_RGB2BGR)
    
    # 5. 画像の表示とユーザーインタラクション
    cv2.imshow('Event Image Visualization', img)
    
    # キーが押されるまで待機 (0は無限待機)。'q'キーで終了。
    key = cv2.waitKey(0)
    if key == ord('q'):
        print("ユーザー操作により終了します。")
        break

# 全てのOpenCVウィンドウを閉じる
cv2.destroyAllWindows()
print("可視化スクリプトが終了しました。")