# gen1/gen1_det.py (修正後)

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import h5py
from .directory import Gen1Directory
from . import io  

class Gen1Det:
    def __init__(self, root: Path, split: str, period_ms: int = 50):
        data_path = root / split
        assert data_path.exists(), f"Data path not found: {data_path}"
        
        self.period_us = period_ms * 1000
        self.directories: Dict[str, Gen1Directory] = {}
        self.reference_timestamps: Dict[str, np.ndarray] = {}
        self.sequence_names: List[str] = []

        h5_files = sorted(data_path.glob("*.dat.h5"))
        for h5_path in h5_files:
            directory = Gen1Directory(h5_path)
            if directory.track_file.exists():
                name = directory.name
                self.directories[name] = directory
                
                with h5py.File(directory.event_file, 'r') as f:
                    events_t = f['events/t']
                    if len(events_t) > 0:
                        t_start, t_end = events_t[0], events_t[-1]
                        self.reference_timestamps[name] = np.arange(t_start, t_end, self.period_us)
                    else:
                        self.reference_timestamps[name] = np.array([])
                
                if len(self.reference_timestamps[name]) > 0:
                    self.sequence_names.append(name)

        self.lengths = [len(self.reference_timestamps[name]) for name in self.sequence_names]
        self.cumulative_lengths = np.cumsum(self.lengths)

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if len(self.cumulative_lengths) > 0 else 0

    def get_item_info(self, global_index: int) -> Tuple[str, int]:
        if not 0 <= global_index < len(self):
            raise IndexError(f"Index {global_index} is out of range.")
        
        seq_list_idx = np.searchsorted(self.cumulative_lengths, global_index, side='right')
        name = self.sequence_names[seq_list_idx]
        
        local_index = global_index
        if seq_list_idx > 0:
            local_index -= self.cumulative_lengths[seq_list_idx - 1]
        
        t_ref = self.reference_timestamps[name][local_index]
        return name, t_ref


    def get_events(self, t_ref: int, directory: Gen1Directory, window_size: int) -> dict:
        t_end = t_ref + window_size
        return io.extract_events_by_timewindow(directory.event_file, t_ref, t_end)

    def get_tracks(self, t_ref: int, directory: Gen1Directory, tolerance_us: int) -> np.ndarray:
        # ラベルは基準時刻t_refの前後±tolerance_usの範囲で検索
        t_start = t_ref - tolerance_us
        t_end = t_ref + tolerance_us
        return io.load_tracks_by_timewindow(directory.track_file, t_start, t_end)