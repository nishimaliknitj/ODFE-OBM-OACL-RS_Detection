"""
Config loader with `_base_` inheritance.

Dataset-specific configs (configs/dior.yaml, etc.) declare `_base_: "default.yaml"`
and only override the keys that differ. This utility merges them recursively.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML config file, resolving `_base_` inheritance if present.

    Parameters
    ----------
    path : str
        Path to the config YAML file (e.g. "configs/dior.yaml").

    Returns
    -------
    dict : fully merged configuration.
    """
    path = os.path.abspath(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Config file {path} is empty or invalid.")

    base_name = cfg.get("_base_")
    if base_name:
        base_path = os.path.join(os.path.dirname(path), base_name)
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def save_config(cfg: Dict[str, Any], path: str) -> None:
    """Persist a resolved config dict to disk (useful for logging exact run config)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def override_from_args(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply CLI overrides like `--no-odfe`, `--no-obm`, `--no-oacl` to a loaded config.
    `overrides` uses dotted keys, e.g. {"odfe.enabled": False}.
    """
    cfg = copy.deepcopy(cfg)
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg
