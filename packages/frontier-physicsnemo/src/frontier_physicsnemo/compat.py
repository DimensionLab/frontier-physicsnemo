# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims between PhysicsNeMo and the environments Frontier runs in.

``torch_scatter``: PhysicsNeMo's MeshGraphNet family and FIGConvNet import ``torch_scatter``
(``scatter`` for edge-to-node aggregation, ``segment_csr`` for neighbourhood reductions) and
check that the distribution is installed. Its wheels are compiled per torch build and do not
exist for every one (none for torch 2.14 at the time of writing; building from source needs a
compiler and the CUDA toolkit). :func:`ensure_torch_scatter` registers a pure-PyTorch module
with the two functions those models call, only when the real package is missing, so the
PhysicsNeMo models themselves run unchanged. The real package is always preferred.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types

import torch

SHIM_VERSION = "2.1.2+frontier.shim"
_REDUCE = {"sum": "sum", "add": "sum", "mean": "mean", "max": "amax", "min": "amin"}


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """``torch_scatter.scatter`` for ``sum | mean | max | min`` with a 1-D ``index`` along
    ``dim`` (the form PhysicsNeMo's graph layers use)."""
    if reduce not in _REDUCE:
        raise ValueError(f"unsupported reduce {reduce!r}")
    dim = dim % src.dim()
    index = index.long()
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0
    shape = list(src.shape)
    shape[dim] = dim_size
    if index.dim() == 1 and src.dim() > 1:
        view = [1] * src.dim()
        view[dim] = -1
        index = index.view(view).expand_as(src)
    if out is None:
        out = src.new_zeros(shape)
        include_self = False
    else:
        include_self = True
    return out.scatter_reduce(dim, index, src, reduce=_REDUCE[reduce], include_self=include_self)


def segment_csr(
    src: torch.Tensor, indptr: torch.Tensor, out: torch.Tensor | None = None, reduce: str = "sum"
) -> torch.Tensor:
    """``torch_scatter.segment_csr`` along the first axis: row ``i`` of the result reduces
    ``src[indptr[i] : indptr[i + 1]]`` (empty segments give 0)."""
    if out is not None:
        raise NotImplementedError("segment_csr(out=...) is not supported by the shim")
    indptr = indptr.long()
    counts = indptr[1:] - indptr[:-1]
    n = counts.shape[0]
    index = torch.repeat_interleave(torch.arange(n, device=src.device), counts)
    return scatter(src, index, dim=0, dim_size=n, reduce=reduce)


def torch_scatter_is_real() -> bool:
    mod = sys.modules.get("torch_scatter")
    if mod is not None:
        return not getattr(mod, "__frontier_shim__", False)
    return importlib.util.find_spec("torch_scatter") is not None


def ensure_torch_scatter() -> str:
    """Make ``import torch_scatter`` work for PhysicsNeMo. Returns ``"native"`` when the real
    package is installed, ``"shim"`` when the pure-PyTorch replacement was registered.

    Must run before the PhysicsNeMo GNN / FIGConvNet modules are imported (their availability
    flags are evaluated at import time); the model adapters of this package call it first.
    """
    if torch_scatter_is_real():
        return "native"
    if "torch_scatter" not in sys.modules:
        shim = types.ModuleType("torch_scatter")
        shim.scatter = scatter  # type: ignore[attr-defined]
        shim.segment_csr = segment_csr  # type: ignore[attr-defined]
        shim.__version__ = SHIM_VERSION  # type: ignore[attr-defined]
        shim.__frontier_shim__ = True  # type: ignore[attr-defined]
        sys.modules["torch_scatter"] = shim
    # PhysicsNeMo asks importlib.metadata whether the distribution exists
    vc = importlib.import_module("physicsnemo.core.version_check")
    original = vc.get_installed_version
    if not getattr(original, "__frontier_shim__", False):

        def get_installed_version(distribution_name: str) -> str | None:
            if distribution_name.replace("-", "_").lower() == "torch_scatter":
                return SHIM_VERSION
            return original(distribution_name)

        get_installed_version.__frontier_shim__ = True  # type: ignore[attr-defined]
        setattr(vc, "get_installed_version", get_installed_version)  # noqa: B010
    stale = [
        name
        for name in sys.modules
        if name.startswith(
            ("physicsnemo.nn.module.gnn_layers", "physicsnemo.models.figconvnet.components")
        )
    ]
    for name in stale:  # imported before the shim existed: their flags say "unavailable"
        del sys.modules[name]
    return "shim"
