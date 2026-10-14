# SPDX-License-Identifier: Apache-2.0
"""Logging bridge from PhysicsNeMo training loops to Frontier runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import frontier


class FrontierLogger:
    """Minimal logger with the same shape as PhysicsNeMo's launch-logging backends.

    Usage inside a PhysicsNeMo example::

        logger = FrontierLogger(project="ahmed-body", name="domino-baseline", config=cfg)
        for step, batch in enumerate(dataloader):
            ...
            logger.log_minibatch({"loss": loss.item()}, step=step)
        logger.finish()
    """

    def __init__(self, project: str, name: str, config: Mapping[str, Any] | None = None) -> None:
        self._run = frontier.init(project=project, name=name, config=_to_plain(config))

    @property
    def run(self) -> frontier.Run:
        return self._run

    def log_minibatch(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._run.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

    def log_epoch(self, metrics: Mapping[str, float], epoch: int) -> None:
        self._run.log({f"epoch/{k}": v for k, v in metrics.items()}, step=epoch)

    def finish(self) -> None:
        self._run.finish()


def _to_plain(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Convert OmegaConf/Hydra configs (used by PhysicsNeMo) to plain dicts if present."""
    if config is None:
        return {}
    try:
        from omegaconf import OmegaConf  # type: ignore[import-not-found]

        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    except ImportError:
        pass
    return dict(config)
