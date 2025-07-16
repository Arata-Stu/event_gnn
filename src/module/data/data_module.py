from omegaconf import DictConfig
import lightning.pytorch as pl
from torch_geometric.data import DataLoader
from typing import Optional

from src.data.dataset.dsec.dataset_for_graph import DSEC
from src.data.dataset.augment import Augmentations

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            augmentations = Augmentations(self.data_cfg.augmentations)
            self.train_dataset = DSEC(root=self.data_cfg.data_path, split="train", transform=augmentations.transform_training, debug=False,
                                min_bbox_diag=15, min_bbox_height=10)
            self.val_dataset = DSEC(root=self.data_cfg.data_path, split="val", transform=augmentations.transform_testing, debug=False,
                                min_bbox_diag=15, min_bbox_height=10)

        elif stage == "val":
            augmentations = Augmentations(self.data_cfg.augmentations)
            self.val_dataset = DSEC(root=self.data_cfg.data_path, split="val", transform=augmentations.transform_testing, debug=False,
                                min_bbox_diag=15, min_bbox_height=10)
        elif stage == "test":
            augmentations = Augmentations(self.data_cfg.augmentations)
            self.test_dataset = DSEC(root=self.data_cfg.data_path, split="test", transform=augmentations.transform_testing, debug=False,
                                min_bbox_diag=15, min_bbox_height=10)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.data_cfg.batch_size.train,
                          shuffle=True,
                          num_workers=self.data_cfg.num_workers.train,
                          follow_batch=['bbox', 'bbox0'],  
                          drop_last=True)                 

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.data_cfg.batch_size.val,
                          shuffle=False,
                          num_workers=self.data_cfg.num_workers.val,
                          follow_batch=['bbox', 'bbox0'], 
                          drop_last=True)                 

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.data_cfg.batch_size.test,
                          shuffle=False,
                          num_workers=self.data_cfg.num_workers.test,
                          follow_batch=['bbox', 'bbox0'], 
                          drop_last=True)                