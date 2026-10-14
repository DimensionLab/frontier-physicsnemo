# SPDX-License-Identifier: Apache-2.0
"""Hydra / OmegaConf compatibility with PhysicsNeMo recipes.

PhysicsNeMo examples describe the network as a Hydra node (``_target_`` plus constructor
arguments, e.g. ``examples/cfd/external_aerodynamics/transformer_models/src/conf/model/
transolver.yaml``). The functions here read such nodes without running Hydra:

- :func:`model_from_hydra` maps a model node to Frontier's ``model: {name, params}``;
- :func:`trainer_config` builds a complete Frontier trainer config around it;
- :func:`native_kwargs` returns the constructor arguments an adapter resolved, in the node's
  own vocabulary, to write a recipe node back (``to_hydra_node``).

Arguments the adapter derives from the dataset (input / output widths) are dropped, and
arguments the adapter does not expose are reported instead of being ignored silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: ``_target_`` -> (registry name, {recipe argument: adapter argument})
TARGETS: dict[str, tuple[str, dict[str, str]]] = {
    "physicsnemo.models.transolver.Transolver": (
        "physicsnemo.transolver",
        {
            "n_layers": "n_layers",
            "n_hidden": "n_hidden",
            "n_head": "n_head",
            "slice_num": "slice_num",
            "mlp_ratio": "mlp_ratio",
            "dropout": "dropout",
            "act": "act",
            "use_te": "use_te",
            "plus": "plus",
        },
    ),
    "physicsnemo.models.geotransolver.GeoTransolver": (
        "physicsnemo.geotransolver",
        {
            "n_layers": "n_layers",
            "n_hidden": "n_hidden",
            "n_head": "n_head",
            "slice_num": "slice_num",
            "mlp_ratio": "mlp_ratio",
            "dropout": "dropout",
            "act": "act",
            "use_te": "use_te",
            "plus": "plus",
            "include_local_features": "include_local_features",
            "radii": "radii",
            "neighbors_in_radius": "neighbors_in_radius",
            "n_hidden_local": "n_hidden_local",
            "attention_type": "attention_type",
        },
    ),
    "physicsnemo.models.figconvnet.FIGConvUNet": (
        "physicsnemo.figconvnet",
        {
            "hidden_channels": "hidden_channels",
            "num_levels": "num_levels",
            "kernel_size": "kernel_size",
            "mlp_channels": "mlp_channels",
            "aabb_min": "aabb_min",
            "aabb_max": "aabb_max",
            "pooling_type": "pooling_type",
            "neighbor_search_type": "neighbor_search_type",
            "knn_k": "knn_k",
        },
    ),
    "physicsnemo.models.meshgraphnet.MeshGraphNet": (
        "physicsnemo.meshgraphnet",
        {
            "processor_size": "processor_size",
            "hidden_dim_processor": "hidden_dim",
            "num_layers_node_processor": "num_layers",
            "aggregation": "aggregation",
            "mlp_activation_fn": "mlp_activation_fn",
        },
    ),
}
#: set by the adapter from the dataset; a value in the recipe is not an error
DERIVED = {
    "functional_dim",
    "out_dim",
    "embedding_dim",
    "geometry_dim",
    "global_dim",
    "in_channels",
    "out_channels",
    "input_dim_nodes",
    "input_dim_edges",
    "output_dim",
    "unified_pos",
    "time_input",
    "structured_shape",
}
_ALIASES = {
    "physicsnemo.models.transolver.transolver.Transolver": (
        "physicsnemo.models.transolver.Transolver"
    ),
    "physicsnemo.models.geotransolver.geotransolver.GeoTransolver": (
        "physicsnemo.models.geotransolver.GeoTransolver"
    ),
    "physicsnemo.models.figconvnet.figconvunet.FIGConvUNet": (
        "physicsnemo.models.figconvnet.FIGConvUNet"
    ),
    "physicsnemo.models.meshgraphnet.meshgraphnet.MeshGraphNet": (
        "physicsnemo.models.meshgraphnet.MeshGraphNet"
    ),
}


def to_plain(node: Any) -> Any:
    """OmegaConf node (or plain container) -> plain Python, interpolations resolved."""
    try:
        from omegaconf import OmegaConf  # type: ignore[import-not-found]

        if OmegaConf.is_config(node):
            return OmegaConf.to_container(node, resolve=True)
    except ImportError:
        pass
    return node


def load_yaml(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Read one recipe YAML with OmegaConf (interpolations, dot-list overrides); falls back
    to PyYAML when OmegaConf is not installed and there are no overrides."""
    try:
        from omegaconf import OmegaConf  # type: ignore[import-not-found]
    except ImportError:
        if overrides:
            raise
        import yaml

        return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    cfg = OmegaConf.load(str(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    plain = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(plain, dict)
    return {str(k): v for k, v in plain.items()}


def model_from_hydra(node: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recipe model node -> ``({"name", "params"}, report)``.

    ``report`` lists ``derived`` (arguments the adapter sets from the dataset, with the
    recipe's value) and ``ignored`` (arguments the adapter does not expose)."""
    plain = to_plain(node)
    if not isinstance(plain, dict) or "_target_" not in plain:
        raise ValueError("a Hydra model node needs a `_target_`")
    target = _ALIASES.get(str(plain["_target_"]), str(plain["_target_"]))
    if target not in TARGETS:
        raise KeyError(f"no Frontier adapter for {target!r}; known: {sorted(TARGETS)}")
    name, mapping = TARGETS[target]
    params: dict[str, Any] = {}
    report: dict[str, Any] = {"target": target, "derived": {}, "ignored": {}}
    for key, value in plain.items():
        if key.startswith("_"):
            continue
        if key in mapping:
            params[mapping[key]] = value
        elif key in DERIVED:
            report["derived"][key] = value
        else:
            report["ignored"][key] = value
    if name == "physicsnemo.transolver" and "out_dim" in plain:
        report["note"] = (
            "out_dim of the recipe counts every target field; "
            "set out_fields to use more than pressure"
        )
    return {"name": name, "params": params}, report


def trainer_config(
    model_node: Any,
    dataset: str,
    *,
    target_field: str = "p",
    point_features: list[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A Frontier trainer config (plain dict) around a recipe's model node."""
    model, _ = model_from_hydra(model_node)
    transient = model["name"] == "physicsnemo.meshgraphnet"
    cfg: dict[str, Any] = {"data": {"path": dataset}, "model": model}
    if transient:
        cfg["task"] = "transient"
    else:
        cfg["data"].update(
            target_field=target_field,
            point_features=list(point_features if point_features is not None else ["normals"]),
        )
    for dotted, value in (overrides or {}).items():
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def to_hydra_node(model: dict[str, Any]) -> dict[str, Any]:
    """Frontier ``model: {name, params}`` -> a recipe-style node with ``_target_``."""
    for target, (name, mapping) in TARGETS.items():
        if name == model.get("name"):
            back = {v: k for k, v in mapping.items()}
            node: dict[str, Any] = {"_target_": target}
            for key, value in (model.get("params") or {}).items():
                if key in back:
                    node[back[key]] = value
            return node
    raise KeyError(f"{model.get('name')!r} is not a PhysicsNeMo adapter")
