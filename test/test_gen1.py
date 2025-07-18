import sys
import cv2
import numpy as np 
from pathlib import Path
from torch_geometric.loader import DataLoader

from src.data.dataset.gen1.dataset_fot_graph import Gen1
from src.visualization.event_viz import draw_events_on_image, map_polarity_to_channel_index

# --- データセットのセットアップ ---
data_path = Path('/media/arata-22/AT_SSD/dataset/gen1').resolve()
split = 'train'

# DSECデータセットの初期化
dataset = Gen1(root=data_path,
               split=split,
               transform=None,
               height=240,
               width=304,
               period_ms=33,
               window_size_ms=1000,
               tolerance_ms=50,
               skip_ts_us=int(5e5),
               min_bbox_diag=30,
               min_bbox_side=10)

# DataLoaderのセットアップ
# バッチサイズを1に設定し、単一のデータインスタンスを処理
batch_size = 1
test_loader = DataLoader(
    dataset, 
    follow_batch=['bbox'], 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=0, 
    drop_last=True
)

print(f"データセットのサイズ: {len(dataset)}")
# --- イベントデータの可視化ループ ---
print("イベントデータの可視化を開始します。")
print("ウィンドウが表示されたら、任意のキーを押すと次の画像へ、'q'キーを押すと終了します。")

for i, data in enumerate(test_loader):
    # 最初の1つのバッチのみを処理（必要に応じて数を増減）
    if i >= 1:
        break
    
    print(f"\n--- Batch {i+1} の処理中 ---")

    ## 空画像を用意 白色で初期化
    img = np.ones((data.height, data.width, 3), dtype=np.uint8) * 255
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