import lightning as pl
from omegaconf import DictConfig
from src.module.data.data_module import DataModule 
from src.module.model.model_module import ModelModule


def fetch_model_module(cfg: DictConfig) -> pl.LightningModule:
    model_str = cfg.model.name
    assert model_str in {'dagr'}, f'Unknown model: {model_str}'
    if model_str == 'dagr':
        return ModelModule(cfg=cfg.model)
    raise NotImplementedError


def fetch_data_module(cfg: DictConfig) -> pl.LightningDataModule:
    data_str = cfg.data.name
    assert data_str in {'dsec'}, f'Unknown data module: {data_str}'
    if data_str == 'dsec':
        return DataModule(cfg=cfg.data)
    raise NotImplementedError
