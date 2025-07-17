"""
This is a modified version of the Pytorch Lightning logger, updated for modern library versions.
"""

import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from weakref import ReferenceType

import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
import wandb
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning.fabric.utilities.logger import (
    _add_prefix,
    _convert_params,
    _flatten_dict,
    _sanitize_callable_params,
)
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

# Version check for PyTorch Lightning
try:
    # Handles versions like '2.3.0', '2.3.0.post0' etc.
    pl_version = pl.__version__.split('.post')[0]
    pl_major_minor = ".".join(pl_version.split('.')[:2])
    pl_is_ge_1_6 = float(pl_major_minor) >= 1.6
except (ValueError, IndexError):
    pl_is_ge_1_6 = False # Fallback for unusual version strings

assert pl_is_ge_1_6, "This logger requires PyTorch Lightning version 1.6 or higher."


class WandbLogger(Logger):
    """
    Custom WandbLogger to log checkpoints as artifacts and manage them.
    """
    LOGGER_JOIN_CHAR = "-"
    STEP_METRIC = "trainer/global_step"

    def __init__(
            self,
            name: Optional[str] = None,
            project: Optional[str] = None,
            group: Optional[str] = None,
            wandb_id: Optional[str] = None,
            prefix: Optional[str] = "",
            log_model: Optional[bool] = True,
            save_last_only_final: Optional[bool] = False,
            config_args: Optional[Union[Dict[str, Any], DictConfig]] = None,
            **kwargs,
    ):
        super().__init__()
        self._experiment: Optional[Run] = None
        self._log_model = log_model
        self._prefix = prefix
        self._logged_model_time: Dict[str, float] = {}
        self._checkpoint_callback: Optional["ReferenceType[ModelCheckpoint]"] = None
        self._save_last: Optional[bool] = None
        self._save_last_only_final = save_last_only_final
        self._config_args = config_args
        self._wandb_init = dict(
            name=name,
            project=project,
            group=group,
            id=wandb_id,
            resume="allow",
            save_code=True,
        )
        self._wandb_init.update(**kwargs)
        self._name = self._wandb_init.get("name")
        self._id = self._wandb_init.get("id")
        self._public_run: Optional[Run] = None

        wandb.require("service")
        _ = self.experiment

    def get_checkpoint(self, artifact_name: str, artifact_filepath: Optional[Path] = None) -> Path:
        """Downloads a checkpoint from a W&B artifact."""
        artifact = self.experiment.use_artifact(artifact_name)
        if artifact_filepath is None:
            assert artifact is not None, 'You are probably using DDP, ' \
                                         'in which case you should provide an artifact filepath.'
            artifact_dir = Path(artifact.download())
            # Find the first .ckpt file in the downloaded directory
            ckpts = list(artifact_dir.glob("*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(f"No .ckpt file found in artifact directory {artifact_dir}")
            artifact_filepath = ckpts[0]

        assert artifact_filepath.exists()
        assert artifact_filepath.suffix == '.ckpt'
        return artifact_filepath

    def __getstate__(self) -> Dict[str, Any]:
        """Handles pickling for distributed training."""
        state = self.__dict__.copy()
        if self._experiment is not None:
            state["_id"] = self._experiment.id
            state["_attach_id"] = getattr(self._experiment, "_attach_id", None)
            state["_name"] = self._experiment.name
        state["_experiment"] = None
        state["_public_run"] = None # Avoid pickling the run object
        return state

    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        """Initializes and returns the W&B run object."""
        if self._experiment is None:
            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                self._experiment = wandb._attach(attach_id)
            else:
                self._experiment = wandb.init(**self._wandb_init)
                if self._config_args is not None:
                    config_dict = OmegaConf.to_container(self._config_args, resolve=True) if isinstance(self._config_args, DictConfig) else self._config_args
                    self._experiment.config.update(config_dict, allow_val_change=True)
                if isinstance(self._experiment, (Run, RunDisabled)) and getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric(self.STEP_METRIC)
                    self._experiment.define_metric("*", step_metric=self.STEP_METRIC, step_sync=True)
        assert isinstance(self._experiment, (Run, RunDisabled))
        return self._experiment

    def watch(self, model: nn.Module, log: str = 'all', log_freq: int = 100, log_graph: bool = True):
        self.experiment.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)
        params = _sanitize_callable_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        if step is not None:
            log_data = {**metrics, self.STEP_METRIC: step}
            self.experiment.log(log_data)
        else:
            self.experiment.log(metrics)

    @property
    def name(self) -> Optional[str]:
        return self._experiment.project if self._experiment else self._name

    @property
    def version(self) -> Optional[str]:
        return self._experiment.id if self._experiment else self._id

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: "ReferenceType[ModelCheckpoint]") -> None:
        if self._checkpoint_callback is None:
            self._checkpoint_callback = checkpoint_callback
            self._save_last = checkpoint_callback.save_last
        if self._log_model:
            self._scan_and_log_checkpoints(checkpoint_callback)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._checkpoint_callback and self._log_model:
            # save_last is always True for finalization
            self._scan_and_log_checkpoints(self._checkpoint_callback, save_last_force=True)

    def _get_public_run(self) -> Run:
        """Gets the public API run object, caching it for efficiency."""
        if self._public_run is None:
            runpath = self.experiment.path
            api = wandb.Api()
            self._public_run = api.run(path=runpath)
        return self._public_run

    def _num_logged_artifact(self) -> int:
        """Returns the number of artifacts logged by the current run."""
        public_run = self._get_public_run()
        return len(list(public_run.logged_artifacts()))

    def _scan_and_log_checkpoints(self, checkpoint_callback: "ReferenceType[ModelCheckpoint]", save_last_force: bool = False) -> None:
        """Scans for new or updated checkpoints and logs them as W&B artifacts."""
        save_last = (self._save_last and not self._save_last_only_final) or save_last_force

        checkpoints = {
            checkpoint_callback.best_model_path: checkpoint_callback.best_model_score,
            **checkpoint_callback.best_k_models,
        }

        if save_last and checkpoint_callback.last_model_path:
            last_model_path = Path(checkpoint_callback.last_model_path)
            if last_model_path.exists():
                checkpoints[str(last_model_path)] = getattr(checkpoint_callback, 'current_score', None)
            else:
                rank_zero_warn(f'Last model checkpoint not found at {checkpoint_callback.last_model_path}')

        valid_checkpoints = []
        for path, score in checkpoints.items():
            if not path: continue
            path_obj = Path(path)
            if path_obj.is_file():
                valid_checkpoints.append((path_obj.stat().st_mtime, str(path), score))
        valid_checkpoints.sort(key=lambda x: x[0])

        new_checkpoints = [
            ckpt for ckpt in valid_checkpoints
            if ckpt[1] not in self._logged_model_time or self._logged_model_time[ckpt[1]] < ckpt[0]
        ]
        new_checkpoints = [x for x in new_checkpoints if x[2] is not None]

        if not new_checkpoints:
            return

        num_ckpt_logged_before = self._num_logged_artifact()

        for time_, path, score in new_checkpoints:
            score_val = score.item() if isinstance(score, torch.Tensor) else score
            metadata = {
                "score": score_val,
                "original_filename": Path(path).name,
                "ModelCheckpoint": {
                    k: getattr(checkpoint_callback, k)
                    for k in ["monitor", "mode", "save_top_k", "save_weights_only"]
                    if hasattr(checkpoint_callback, k)
                },
            }
            aliases = []
            if path == checkpoint_callback.best_model_path:
                aliases.append('best')
            if path == checkpoint_callback.last_model_path:
                aliases.append('last')

            artifact_name = f"{self.experiment.name}-{self.experiment.id}"
            artifact = wandb.Artifact(name=artifact_name, type='model', metadata=metadata)
            artifact.add_file(path, name="model.ckpt")
            self.experiment.log_artifact(artifact, aliases=aliases)
            self._logged_model_time[path] = time_

        timeout = 30
        time_spent = 0
        while self._num_logged_artifact() < num_ckpt_logged_before + len(new_checkpoints):
            time.sleep(1)
            time_spent += 1
            if time_spent >= timeout:
                rank_zero_warn("Timeout: W&B artifacts did not finish uploading in time.")
                break

        if checkpoint_callback.save_top_k > 0:
            try:
                self._rm_but_top_k(checkpoint_callback.save_top_k)
            except Exception as e:
                rank_zero_warn(f"Failed to remove old artifacts: {e}")

    def _rm_but_top_k(self, top_k: int):
        """Removes artifacts to enforce the save_top_k limit."""
        public_run = self._get_public_run()
        artifacts = list(public_run.logged_artifacts())

        candidates_for_deletion = []
        for art in artifacts:
            if art.type != 'model' or 'best' in art.aliases or 'last' in art.aliases:
                continue

            score = art.metadata.get("score")
            if score is None or (isinstance(score, (int, float)) and np.isinf(score)):
                continue

            candidates_for_deletion.append(art)

        candidates_for_deletion.sort(key=lambda art: art.metadata["score"], reverse=True)

        for art_to_delete in candidates_for_deletion[top_k:]:
            try:
                art_to_delete.delete(delete_aliases=True)
                rank_zero_warn(f"Deleted old artifact {art_to_delete.name} with score {art_to_delete.metadata['score']}")
            except wandb.errors.CommError as e:
                rank_zero_warn(f"Failed to delete artifact {art_to_delete.name} due to a communication error: {e}")


def get_wandb_logger(full_config: DictConfig) -> WandbLogger:
    """Initializes the WandbLogger from a DictConfig."""
    wandb_config = full_config.wandb
    wandb_runpath = wandb_config.get("wandb_runpath")

    if wandb_runpath is None:
        wandb_id = wandb.util.generate_id()
        print(f'New run: generating id {wandb_id}')
    else:
        wandb_id = Path(wandb_runpath).name
        print(f'Using provided id {wandb_id}')

    logger = WandbLogger(
        project=wandb_config.project_name,
        group=wandb_config.get("group_name"),
        wandb_id=wandb_id,
        log_model=True,
        save_last_only_final=False,
        config_args=full_config,
    )

    return logger


def get_ckpt_path(logger: WandbLogger, wandb_config: DictConfig) -> Optional[Path]:
    """Gets a local checkpoint path from a W&B artifact."""
    artifact_name = wandb_config.get("artifact_name")
    if not artifact_name:
        return None

    print(f'Resuming checkpoint from artifact {artifact_name}')
    artifact_local_file = wandb_config.get("artifact_local_file")
    if artifact_local_file:
        artifact_local_file = Path(artifact_local_file)

    resume_path = logger.get_checkpoint(
        artifact_name=artifact_name,
        artifact_filepath=artifact_local_file
    )

    assert resume_path.exists() and resume_path.suffix == '.ckpt'
    return resume_path