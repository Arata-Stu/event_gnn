import torch
import math
from copy import deepcopy
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from src.model.networks.ema import ModelEMA

class EMACallback(Callback):
    """
    カスタムModelEMAクラスをPyTorch Lightningで利用するためのCallback。
    """
    def __init__(self, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.ema_model = None

    def on_fit_start(self, trainer, pl_module):
        """学習開始時にEMAモデルを初期化する。"""
        if self.ema_model is None:
            self.ema_model = ModelEMA(pl_module, decay=self.decay)
            self.ema_model.ema.to(device=pl_module.device)
        print("EMACallback: EMA model initialized.")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """各学習バッチの終了後にEMAの重みを更新する。"""
        self.ema_model.update(pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        """検証開始前に、モデルの重みをEMAの重みと入れ替える。"""
        # 元の重みを保存
        self.original_state_dict = deepcopy(pl_module.state_dict())
        # EMAの重みをモデルにコピー
        pl_module.load_state_dict(self.ema_model.ema.state_dict(), strict=True)
        print("EMACallback: Swapped to EMA weights for validation.")

    def on_validation_epoch_end(self, trainer, pl_module):
        """検証終了後に、モデルの重みを元の学習中の重みに戻す。"""
        pl_module.load_state_dict(self.original_state_dict, strict=True)
        del self.original_state_dict # メモリを解放
        print("EMACallback: Restored original weights after validation.")
        
    def on_test_epoch_start(self, trainer, pl_module):
        """テスト開始前に、モデルの重みをEMAの重みと入れ替える。"""
        self.original_state_dict = deepcopy(pl_module.state_dict())
        pl_module.load_state_dict(self.ema_model.ema.state_dict(), strict=True)
        print("EMACallback: Swapped to EMA weights for testing.")

    def on_test_epoch_end(self, trainer, pl_module):
        """テスト終了後に、モデルの重みを元の学習中の重みに戻す。"""
        pl_module.load_state_dict(self.original_state_dict, strict=True)
        del self.original_state_dict
        print("EMACallback: Restored original weights after testing.")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """チェックポイントにEMAの状態を保存する。"""
        if self.ema_model:
            checkpoint['ema_state_dict'] = self.ema_model.ema.state_dict()
            checkpoint['ema_updates'] = self.ema_model.updates

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """チェックポイントからEMAの状態を復元する。"""
        if 'ema_state_dict' in checkpoint:
            self.ema_model = ModelEMA(
                pl_module, 
                decay=self.decay, 
                updates=checkpoint['ema_updates']
            )
            self.ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
            self.ema_model.ema.to(device=pl_module.device)
            print("EMACallback: EMA model state loaded from checkpoint.")