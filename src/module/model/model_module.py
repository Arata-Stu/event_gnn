from typing import Optional

import torch
import lightning.pytorch as pl
from omegaconf import DictConfig
from typing import Any, Optional

from src.utils.gradients import fix_gradients
from src.utils.data_utils import format_data
from src.model.utils import convert_to_training_format, postprocess_network_output, convert_to_evaluation_format
from src.model.networks.dagr import DAGR

class ModelModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg
        height = 215
        width = 320
        self.model = DAGR(cfg.model, height=height, width=width)

    def setup(self, stage: Optional[str] = None) -> None:
        self.started_training = True
        if stage == "fit":
            self.train_cfg = self.cfg.training
        elif stage == "val":
            pass
        elif stage == "test":
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, data, batch_idx):
        self.started_training = True
        data = format_data(data)
        targets = convert_to_training_format(data.bbox, data.bbox_batch, data.num_graphs)
        if self.model.backbone.use_image:
            targets0 = convert_to_training_format(data.bbox0, data.bbox0_batch, data.num_graphs)
            targets = (targets, targets0)

        outputs = self.model(data, reset=True, targets=targets)
        loss_dict = {k: v for k, v in outputs.items() if "loss" in k}
        loss = loss_dict.pop("total_loss")
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, data, batch_idx):
        data = format_data(data)
        predictions = self.model(data, reset=True)
        detections = postprocess_network_output(predictions, self.backbone.num_classes, self.conf_threshold, self.nms_threshold, filtering=filtering,
                                                height=self.height, width=self.width)

        ret = [detections]

        if hasattr(data, 'bbox'):
            targets = convert_to_evaluation_format(data)

        loss = None
        return loss

    def test_step(self, data, batch_idx):
        data = format_data(data)
        outputs = self.model(data, reset=True)
        detections = postprocess_network_output(outputs, self.backbone.num_classes, self.conf_threshold, self.nms_threshold, filtering=filtering,
                                                height=self.height, width=self.width)

        ret = [detections]

        if hasattr(data, 'bbox'):
            targets = convert_to_evaluation_format(data)

        loss = None
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        # 勾配にnanが含まれていれば0に修正する
        fix_gradients(self)

    def configure_optimizers(self) -> Any:
        lr = self.train_cfg.learning_rate
        weight_decay = self.train_cfg.weight_decay
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.train_cfg.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0
        # Here we interpret the final lr as max_lr/final_div_factor.
        # Note that Pytorch OneCycleLR interprets it as initial_lr/final_div_factor:
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
