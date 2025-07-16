import torch
import lightning.pytorch as pl

class EmptyCacheCallback(pl.Callback):
    """
    バリデーションとテストの各バッチ終了時に
    torch.cuda.empty_cache() を呼び出すコールバック。
    """
    def __init__(self):
        super().__init__()
        print("✅ Initialized EmptyCacheCallback: Cuda cache will be cleared after each validation and test step.")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """バリデーションの各バッチ処理後に呼び出される"""
        torch.cuda.empty_cache()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """テストの各バッチ処理後に呼び出される"""
        torch.cuda.empty_cache()