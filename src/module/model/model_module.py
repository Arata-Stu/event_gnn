import torch
import lightning.pytorch as pl
from omegaconf import DictConfig
from typing import Any

from src.utils.gradients import fix_gradients
from src.utils.data_utils import format_data
from src.model.utils import convert_to_training_format, postprocess_network_output, convert_to_evaluation_format
from src.model.networks.dagr import DAGR
from src.utils.buffers import DetectionBuffer
from src.utils.data_utils import dataset_2_hw, dataset_2_classes

class ModelModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg
        self.height, self.width = dataset_2_hw[cfg.data.name]
        self.classes = dataset_2_classes[cfg.data.name]
        self.model = DAGR(cfg.model, height=self.height, width=self.width)

        self.buffer = DetectionBuffer(height=self.height, width=self.width, classes=self.classes)

        self.save_hyperparameters(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, data, batch_idx):
        data = format_data(data)
        correct_batch_size = data.num_graphs
        
        targets = convert_to_training_format(data.bbox, data.bbox_batch, correct_batch_size)
        if self.model.backbone.use_image:
            targets0 = convert_to_training_format(data.bbox0, data.bbox0_batch, correct_batch_size)
            targets = (targets, targets0)

        outputs = self.model(data, reset=True, targets=targets)
        loss_dict = {f"train/{k}": v for k, v in outputs.items() if "loss" in k}
        loss = loss_dict.pop("train/loss")
        
        self.log_dict(loss_dict, 
                      on_step=True, 
                      on_epoch=True, 
                      prog_bar=False, 
                      logger=True,
                      batch_size=correct_batch_size) 
                      
        self.log("train_loss", 
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=correct_batch_size) 

        return loss

    def validation_step(self, data, batch_idx):
        """各バッチで予測を行い、結果をバッファに蓄積"""
        data = format_data(data)
        
        # 予測の実行
        predictions = self.model(data, reset=True)
        detections = postprocess_network_output(
            predictions, 
            self.model.backbone.num_classes, 
            self.model.conf_threshold, 
            self.model.nms_threshold, 
            filtering=True,
            height=self.height, 
            width=self.width
        )

        # 正解ラベルの取得
        if hasattr(data, 'bbox'):
            targets = convert_to_evaluation_format(data)
            # バッファを更新
            self.buffer.update(detections, targets)
        

    def on_validation_epoch_end(self):
        """バリデーションエポック終了時に評価指標を計算し、ログに記録"""
        metrics = self.buffer.compute()
        log_metrics = {f"val/{k}": v for k, v in metrics.items()}
        self.log_dict(log_metrics, prog_bar=True, logger=True)

    def test_step(self, data, batch_idx):
        """テストステップでの予測とバッファ更新"""
        data = format_data(data)
        outputs = self.model(data, reset=True)
        detections = postprocess_network_output(
            outputs, 
            self.model.backbone.num_classes, 
            self.model.conf_threshold, 
            self.model.nms_threshold, 
            filtering=True,
            height=self.height, 
            width=self.width
        )
        
        if hasattr(data, 'bbox'):
            targets = convert_to_evaluation_format(data)
            self.buffer.update(detections, targets) 
            
    def on_test_epoch_end(self):
        """テストエポック終了時に評価指標を計算し、ログに記録"""
        metrics = self.buffer.compute()
        log_metrics = {f"test/{k}": v for k, v in metrics.items()}
        self.log_dict(log_metrics, prog_bar=True, logger=True)
    
    def on_before_optimizer_step(self, optimizer):
        fix_gradients(self)

    def configure_optimizers(self) -> Any:
        lr = self.cfg.training.learning_rate
        weight_decay = self.cfg.training.weight_decay
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.cfg.training.lr_scheduler
        if not scheduler_params.use:
            return optimizer
        
        total_steps = self.trainer.estimated_stepping_batches
        
        final_div_factor_pytorch = scheduler_params.final_div_factor / scheduler_params.div_factor
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy='linear')
        
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}