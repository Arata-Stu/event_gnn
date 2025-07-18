from pathlib import Path

class Gen1Directory:
    
    def __init__(self, h5_path: Path):
        self.root: Path = h5_path.parent
        self.name: str = h5_path.name.removesuffix('.dat.h5')
        self.event_file: Path = h5_path
        self.track_file: Path = self.root / f"{self.name}_bbox.npy"