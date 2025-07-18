# gen1/dataset.py (最終調整版)

from pathlib import Path
from typing import Callable, Optional
import torch
import numpy as np
from torch_geometric.data import Dataset

from .dataset import Gen1Det
from .utils import tracks_to_array, filter_boxes
from src.utils.data_utils import to_data

class Gen1(Dataset):
    """
    DSECクラスの設計を模倣した、Gen1データセット用の高レベルインターフェース。
    """
    def __init__(self,
                 root: Path,
                 split: str,
                 transform: Optional[Callable] = None,
                 height: int = 240,
                 width: int = 304,
                 period_ms: int = 50,
                 window_size_ms: int = 1000,
                 tolerance_ms: int = 50,
                 skip_ts_us: int = int(5e5),
                 min_bbox_diag: int = 30,
                 min_bbox_side: int = 10):
        super().__init__()
        self.dataset_det = Gen1Det(root, split, period_ms=period_ms)
        self.height = height
        self.width = width
        self.time_window = window_size_ms * 1000
        self.tolerance_us = tolerance_ms * 1000
        self.transform = transform
        self.skip_ts_us = skip_ts_us
        self.min_bbox_diag = min_bbox_diag
        self.min_bbox_side = min_bbox_side

    def __len__(self) -> int:
        return len(self.dataset_det)
    
    def __getitem__(self, idx: int):
        # 1. 生データを取得
        name, t_ref = self.dataset_det.get_item_info(idx)
        directory = self.dataset_det.directories[name]
        events = self.dataset_det.get_events(t_ref, directory, self.time_window)
        tracks = self.dataset_det.get_tracks(t_ref, directory, self.tolerance_us)

        # 2. 前処理
        seq_start_ts = self.dataset_det.reference_timestamps[name][0]
        tracks = self.preprocess_tracks(tracks, sequence_start_ts=seq_start_ts)
        events = self.preprocess_events(events)
        
        # 3. to_data関数に渡すためのフラットな辞書を作成
        data_dict = {**events}
        
        data_dict['bbox'] = tracks_to_array(tracks)
        data_dict['seq_name'] = name
        data_dict['timestamp'] = t_ref
        data_dict['height'] = self.height
        data_dict['width'] = self.width
        data_dict['time_window'] = self.time_window
        
        data = to_data(**data_dict)

        # 5. データ拡張
        if self.transform is not None:
            data = self.transform(data)
            
        return data

    def preprocess_tracks(self, tracks: np.ndarray, sequence_start_ts: int) -> np.ndarray:
        return filter_boxes(
            tracks=tracks,
            skip_ts=self.skip_ts_us,
            min_box_diag=self.min_bbox_diag,
            min_box_side=self.min_bbox_side,
            sequence_start_ts=sequence_start_ts
        )

    def preprocess_events(self, events: dict) -> dict:
        if len(events['t']) > 0:
            events['t'] = self.time_window + (events['t'] - events['t'][-1])
        events['p'] = 2 * events['p'].astype(np.int8) - 1
        return events