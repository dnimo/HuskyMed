import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")


@dataclass
class HuskyConfig:
    raw: Dict[str, Any] = field(default_factory=dict)

    def get(self, path: str, default: Any = None) -> Any:
        # dot-path getter, e.g., get("data_paths.input")
        cur = self.raw
        for part in path.split('.'):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def ensure_dirs(self) -> None:
        # create standard directories if exist in config
        for p in [
            self.get("data_paths.input"),
            self.get("data_paths.output"),
            self.get("data_paths.processed"),
            self.get("data_paths.test"),
            self.get("model_paths.model_directory"),
            self.get("model_paths.pretrained_model"),
            self.get("model_paths.peft_model"),
            self.get("log_paths.log_directory"),
        ]:
            if isinstance(p, str):
                os.makedirs(p, exist_ok=True)


_CFG: HuskyConfig | None = None


def load_config(config_path: str | None = None, overrides: Dict[str, Any] | None = None) -> HuskyConfig:
    """
    Load config.json and apply optional overrides.
    """
    global _CFG
    if _CFG is not None:
        return _CFG

    path = config_path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        data = json.load(f)

    if overrides:
        data = _deep_merge_dicts(data, overrides)

    _CFG = HuskyConfig(raw=data)
    _CFG.ensure_dirs()
    return _CFG


def get_config() -> HuskyConfig:
    if _CFG is None:
        return load_config()
    return _CFG


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out
