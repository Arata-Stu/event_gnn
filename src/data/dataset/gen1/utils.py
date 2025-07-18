import numpy as np

def tracks_to_array(tracks: np.ndarray) -> np.ndarray:
    """Numpyの構造化配列から、(x, y, w, h, class_id)の形式のTensorに変換する。"""
    if len(tracks) == 0:
        return np.empty((0, 5), dtype=np.float32)
    
    # Gen1データセットのトラックのフィールド名に合わせて調整してください
    return np.stack([
        tracks['x'], tracks['y'], tracks['w'], tracks['h'], tracks['class_id']
    ], axis=1)

def filter_boxes(tracks: np.ndarray, 
                 skip_ts: int = int(5e5), 
                 min_box_diag: int = 30, 
                 min_box_side: int = 10,
                 sequence_start_ts: int = 0) -> np.ndarray:
    
    if len(tracks) == 0:
        return tracks

    ts = tracks['t']
    width = tracks['w']
    height = tracks['h']
    
    diag_square = width**2 + height**2
    
    # skip_tsをシーケンスごとの相対的な期間として扱う
    absolute_skip_ts = sequence_start_ts + skip_ts
    
    # 全ての条件を結合したマスクを作成
    mask = (ts > absolute_skip_ts) & \
           (diag_square >= min_box_diag**2) & \
           (width >= min_box_side) & \
           (height >= min_box_side)
    
    return tracks[mask]
