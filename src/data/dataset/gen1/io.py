# gen1/io.py

import h5py
import numpy as np
from pathlib import Path

def get_num_events(h5_file: Path) -> int:
    """
    h5ファイル内のイベント総数を取得します。
    """
    with h5py.File(str(h5_file), 'r') as f:
        return len(f['events/t'])

def extract_events_by_timewindow(h5_file: Path, t_start_us: int, t_end_us: int) -> dict:
    """
    開始・終了時刻(μs)に基づいてh5ファイルからイベントを抽出します。
    
    Gen1データセットには'ms_to_idx'のような高速な参照テーブルがないため、
    イベントのタイムスタンプ't'配列に対して直接バイナリサーチ(np.searchsorted)を
    実行し、目的のイベント範囲を効率的に特定します。
    """
    with h5py.File(str(h5_file), 'r') as f:
        events_t = f['events/t']
        
        # バイナリサーチで時間窓に対応するイベントの開始・終了インデックスを特定
        start_idx = np.searchsorted(events_t, t_start_us, side='left')
        end_idx = np.searchsorted(events_t, t_end_us, side='right')
        
        return {
            't': events_t[start_idx:end_idx],
            'x': f['events/x'][start_idx:end_idx],
            'y': f['events/y'][start_idx:end_idx],
            'p': f['events/p'][start_idx:end_idx],
        }

def load_tracks_by_timewindow(track_file: Path, t_start_us: int, t_end_us: int) -> np.ndarray:
    """
    開始・終了時刻(μs)に基づいてnpyファイルからトラック(ラベル)をロードします。
    """
    # メモリマップモードでファイルを開き、必要な部分だけを効率的に読み込みます
    all_tracks = np.load(track_file, mmap_mode='r')
    
    # タイムスタンプが指定範囲内にあるトラックを検索します
    labels_t = all_tracks['t']
    indices = np.where((labels_t >= t_start_us) & (labels_t < t_end_us))
    
    # メモリマップされた配列から実際のデータをコピーして返します
    return all_tracks[indices].copy()