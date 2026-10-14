# SPDX-License-Identifier: Apache-2.0
"""NVIDIA PhysicsNeMo integration for Frontier.

- ``frontier_physicsnemo.checkpoints``: ``.mdlus`` next to Frontier checkpoints;
- ``frontier_physicsnemo.hydra``: recipe (Hydra) model nodes <-> Frontier configs;
- ``frontier_physicsnemo.compat``: pure-PyTorch ``torch_scatter`` where no wheel exists;
- ``FrontierLogger``: logging backend for PhysicsNeMo training loops.

The model adapters (``model.name: physicsnemo.*``), datapipes and pretrained loaders are
``frontier_train.models.physicsnemo_ref``, ``frontier_train.physicsnemo_datapipes`` and
``frontier_train.pretrained``; they build on this package. Only the logger is imported here,
so the package imports without torch or PhysicsNeMo.
"""

from frontier_physicsnemo.logging import FrontierLogger

__all__ = ["FrontierLogger"]
