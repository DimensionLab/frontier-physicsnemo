# SPDX-License-Identifier: Apache-2.0
"""The PhysicsNeMo bridge: scatter shim, .mdlus reader, Hydra mapping. No trainer, no GPU."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
import torch

from frontier_physicsnemo import compat
from frontier_physicsnemo.checkpoints import read_mdlus
from frontier_physicsnemo.hydra import model_from_hydra, to_hydra_node, trainer_config

HAS_PHYSICSNEMO = compat.importlib.util.find_spec("physicsnemo") is not None
needs_physicsnemo = pytest.mark.skipif(not HAS_PHYSICSNEMO, reason="nvidia-physicsnemo missing")


def test_scatter_shim_matches_reference():
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    index = torch.tensor([0, 2, 0, 2])
    out = compat.scatter(src, index, dim=0, dim_size=3, reduce="sum")
    assert torch.equal(out, torch.tensor([[6.0, 8.0], [0.0, 0.0], [10.0, 12.0]]))
    mean = compat.scatter(src, index, dim=0, dim_size=3, reduce="mean")
    assert torch.equal(mean[0], torch.tensor([3.0, 4.0]))
    mx = compat.scatter(src, index, dim=0, dim_size=3, reduce="max")
    assert torch.equal(mx[2], torch.tensor([7.0, 8.0]))
    seg = compat.segment_csr(src, torch.tensor([0, 1, 1, 4]), reduce="mean")
    assert torch.equal(seg, torch.tensor([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]]))
    with pytest.raises(ValueError):
        compat.scatter(src, index, reduce="prod")


def _zip_mdlus(path: Path, state: dict[str, torch.Tensor]) -> None:
    buf = io.BytesIO()
    torch.save(state, buf)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("model.pt", buf.getvalue())
        zf.writestr("args.json", json.dumps({"__name__": "Tiny", "__args__": {"width": 2}}))
        zf.writestr("metadata.json", json.dumps({"mdlus_file_version": "0.2.0"}))


def test_read_mdlus_layouts(tmp_path):
    state = {"layer.weight": torch.ones(2, 2), "layer.bias": torch.zeros(2)}
    _zip_mdlus(tmp_path / "a.mdlus", state)
    got, meta = read_mdlus(tmp_path / "a.mdlus")
    assert meta["layout"] == "mdlus-zip" and meta["args"]["__args__"] == {"width": 2}
    assert torch.equal(got["layer.weight"], state["layer.weight"])
    # published checkpoints that are plain torch.save archives with the .mdlus extension
    torch.save({"model": state, "epoch": 3}, tmp_path / "b.mdlus")
    got, meta = read_mdlus(tmp_path / "b.mdlus")
    assert meta["layout"] == "torch-archive" and set(got) == set(state)


def test_hydra_nodes_round_trip():
    node = {
        "_target_": "physicsnemo.models.transolver.Transolver",
        "functional_dim": 2,
        "out_dim": 4,
        "embedding_dim": 6,
        "n_layers": 8,
        "n_hidden": 256,
        "n_head": 8,
        "slice_num": 512,
        "mlp_ratio": 2,
        "unified_pos": False,
        "ref": 8,
    }
    model, report = model_from_hydra(node)
    assert model == {
        "name": "physicsnemo.transolver",
        "params": {"n_layers": 8, "n_hidden": 256, "n_head": 8, "slice_num": 512, "mlp_ratio": 2},
    }
    assert report["derived"]["out_dim"] == 4 and report["ignored"] == {"ref": 8}
    back = to_hydra_node(model)
    assert back["_target_"] == node["_target_"] and back["slice_num"] == 512
    cfg = trainer_config(node, "datasets/x", target_field="pMean", overrides={"train.epochs": 3})
    assert cfg["data"]["point_features"] == ["normals"] and cfg["train"]["epochs"] == 3
    mgn = trainer_config(
        {"_target_": "physicsnemo.models.meshgraphnet.MeshGraphNet", "hidden_dim_processor": 64},
        "datasets/cyl",
    )
    assert mgn["task"] == "transient" and mgn["model"]["params"] == {"hidden_dim": 64}
    with pytest.raises(KeyError, match="no Frontier adapter"):
        model_from_hydra({"_target_": "physicsnemo.models.fno.FNO"})


# -- with PhysicsNeMo ------------------------------------------------------------------------------
