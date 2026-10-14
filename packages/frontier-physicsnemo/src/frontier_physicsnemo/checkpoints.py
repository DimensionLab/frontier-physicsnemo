# SPDX-License-Identifier: Apache-2.0
"""PhysicsNeMo ``.mdlus`` checkpoints next to Frontier's own.

A Frontier checkpoint (``best.pt``) holds the adapter's state dict plus the model card. The
wrapped PhysicsNeMo network is additionally saved as ``best.mdlus`` with
``physicsnemo.Module.save`` so it loads in PhysicsNeMo itself
(``physicsnemo.Module.from_checkpoint("best.mdlus")``), in a NIM-style server or in one of
their recipes, without Frontier.

Reading needs no PhysicsNeMo: a ``.mdlus`` file is a zip (older ones a tar) holding
``model.pt``, ``args.json`` and ``metadata.json``; some published checkpoints carry the
extension but are plain ``torch.save`` archives. :func:`read_mdlus` handles all three and
returns the state dict with the constructor arguments when the file has them.

These are third-party files, usually downloaded. Their tensors are read with
``torch.load(..., weights_only=True)`` and nothing else: a file whose pickle holds more than
tensors and plain containers is refused with :class:`UnsafeMdlusError`, and there is no flag
to load it anyway. :func:`load_mdlus`, which hands the file to
PhysicsNeMo, first applies the same check and requires the class named in ``args.json`` to
live under ``physicsnemo.``.
"""

from __future__ import annotations

import io
import json
import pickle
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_mdlus(module: nn.Module, path: str | Path) -> Path:
    """Save a PhysicsNeMo ``Module`` to ``path`` (``.mdlus``)."""
    path = Path(path)
    if path.suffix != ".mdlus":
        path = path.with_suffix(".mdlus")
    save = getattr(module, "save", None)
    if save is None or not hasattr(module, "meta"):
        raise TypeError(f"{type(module).__name__} is not a physicsnemo.Module")
    path.parent.mkdir(parents=True, exist_ok=True)
    save(str(path))
    return path


#: module prefixes the class named in a ``.mdlus`` file's ``args.json`` may come from
ALLOWED_MDLUS_MODULE_PREFIXES: tuple[str, ...] = ("physicsnemo.",)


class UnsafeMdlusError(RuntimeError):
    """A ``.mdlus`` / third-party checkpoint that cannot be read as plain tensors."""


def load_mdlus(path: str | Path, **override_args: Any) -> nn.Module:
    """Instantiate the PhysicsNeMo model stored in a ``.mdlus`` file (needs PhysicsNeMo).

    PhysicsNeMo imports the class named in the file's ``args.json`` and loads ``model.pt``
    itself, so the file is checked first: its tensors must load under ``weights_only=True``
    and the class must live under :data:`ALLOWED_MDLUS_MODULE_PREFIXES`."""
    _, meta = read_mdlus(path)
    module = str((meta.get("args") or {}).get("__module__") or "")
    if not module or not f"{module}.".startswith(ALLOWED_MDLUS_MODULE_PREFIXES):
        raise UnsafeMdlusError(
            f"{path}: the checkpoint names the class module {module!r}; only classes under "
            f"{', '.join(ALLOWED_MDLUS_MODULE_PREFIXES)} are instantiated from a .mdlus file"
        )
    import physicsnemo  # type: ignore[import-not-found]

    return physicsnemo.Module.from_checkpoint(str(path), override_args=override_args or None)


def _load_state(buffer: bytes, name: str = "checkpoint") -> Any:
    """Tensors and plain containers only; there is deliberately no unsafe retry."""
    try:
        return torch.load(io.BytesIO(buffer), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is the unpickler's refusal
        text = str(exc)
        refused = isinstance(exc, pickle.UnpicklingError) or (
            isinstance(exc, RuntimeError) and ("weights_only" in text or "TorchScript" in text)
        )
        if not refused:
            raise
        raise UnsafeMdlusError(
            f"{name}: this checkpoint contains pickled Python objects; loading it can execute "
            "code. Third-party checkpoints are only read as plain tensors (weights_only=True) "
            "and --trust-checkpoint does not apply to them; ask the publisher for a "
            "state-dict or safetensors file"
        ) from exc


def read_mdlus(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """State dict and metadata (``args``, ``metadata``, ``layout``) of a ``.mdlus`` file."""
    path = Path(path)
    meta: dict[str, Any] = {"file": path.name}
    raw: Any = None
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if "model.pt" in names:
                raw = _load_state(archive.read("model.pt"), str(path))
                meta["layout"] = "mdlus-zip"
                for key, name in (("args", "args.json"), ("metadata", "metadata.json")):
                    if name in names:
                        meta[key] = json.loads(archive.read(name))
        if raw is None:  # a torch.save archive (also a zip) with the .mdlus extension
            raw = _load_state(path.read_bytes(), str(path))
            meta["layout"] = "torch-archive"
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as tar:
            members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
            if "model.pt" not in members:
                raise ValueError(f"{path}: tar archive without model.pt")
            fh = tar.extractfile(members["model.pt"])
            assert fh is not None
            raw = _load_state(fh.read(), str(path))
            meta["layout"] = "mdlus-tar"
            for key, name in (("args", "args.json"), ("metadata", "metadata.json")):
                if name in members:
                    fj = tar.extractfile(members[name])
                    assert fj is not None
                    meta[key] = json.loads(fj.read())
    else:
        raw = _load_state(path.read_bytes(), str(path))
        meta["layout"] = "torch-pickle"
    state = _find_state(raw)
    return state, meta


def _find_state(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, nn.Module):
        return dict(obj.state_dict())
    if isinstance(obj, dict):
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return dict(obj)
        for key in ("model", "state_dict", "model_state_dict"):
            inner = obj.get(key)
            if (
                isinstance(inner, dict)
                and inner
                and all(torch.is_tensor(v) for v in inner.values())
            ):
                return dict(inner)
        tensors = {k: v for k, v in obj.items() if torch.is_tensor(v)}
        if tensors:
            return tensors
    raise TypeError(f"no state dict in the checkpoint (got {type(obj).__name__})")
