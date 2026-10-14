# SPDX-License-Identifier: Apache-2.0
"""Security review DATA-01: third-party checkpoints are read as plain tensors or not at all."""

from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch

from frontier_physicsnemo import checkpoints
from frontier_physicsnemo.checkpoints import UnsafeMdlusError, load_mdlus, read_mdlus


class _Evil:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (open, (str(self.marker), "w"))


def _pickle(marker: Path) -> bytes:
    buf = io.BytesIO()
    torch.save({"model": {"w": torch.zeros(2)}, "boom": _Evil(marker)}, buf)
    return buf.getvalue()


def _write(path: Path, layout: str, model_pt: bytes, args: dict[str, Any] | None = None) -> Path:
    members = {"model.pt": model_pt, "args.json": json.dumps(args or {}).encode()}
    if layout == "pickle":
        path.write_bytes(model_pt)
    elif layout == "zip":
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
    else:
        with tarfile.open(path, "w") as tar:
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return path


@pytest.mark.parametrize("layout", ["pickle", "zip", "tar"])
def test_pickled_objects_in_a_third_party_checkpoint_are_never_executed(tmp_path, layout):
    marker = tmp_path / "executed.marker"
    path = _write(tmp_path / "evil.mdlus", layout, _pickle(marker))
    with pytest.raises(UnsafeMdlusError, match="does not apply"):
        read_mdlus(path)
    with pytest.raises(UnsafeMdlusError):
        load_mdlus(path)
    assert not marker.exists()


def test_there_is_no_unsafe_load_in_the_reader(monkeypatch):
    seen: list[Any] = []
    real = torch.load

    def spy(*a: Any, **k: Any) -> Any:
        seen.append(k.get("weights_only"))
        return real(*a, **k)

    monkeypatch.setattr(checkpoints.torch, "load", spy)
    buf = io.BytesIO()
    torch.save({"w": torch.ones(1)}, buf)
    assert set(checkpoints._load_state(buf.getvalue())) == {"w"}
    with pytest.raises(UnsafeMdlusError):
        checkpoints._load_state(_pickle(Path("never-created.marker")))
    assert seen == [True, True] and not Path("never-created.marker").exists()


def test_corrupt_files_keep_their_own_error(tmp_path):
    (tmp_path / "junk.mdlus").write_bytes(b"not a checkpoint")
    with pytest.raises(Exception) as err:  # noqa: PT011 - whatever torch raises, not ours
        read_mdlus(tmp_path / "junk.mdlus")
    assert not isinstance(err.value, UnsafeMdlusError) or "pickled" in str(err.value)


@pytest.mark.parametrize("module", ["subprocess", "os", "physicsnemo_evil.models", "", None])
def test_load_mdlus_only_instantiates_physicsnemo_classes(tmp_path, module, monkeypatch):
    buf = io.BytesIO()
    torch.save({"w": torch.ones(1)}, buf)
    args = {"__name__": "Popen", "__module__": module, "__args__": {"args": "true"}}
    path = _write(tmp_path / "m.mdlus", "zip", buf.getvalue(), args)
    monkeypatch.setitem(__import__("sys").modules, "physicsnemo", None)  # must not be reached
    with pytest.raises(UnsafeMdlusError, match="only classes under"):
        load_mdlus(path)
