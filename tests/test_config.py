import os
from config import load_config

def test_config_load_and_dirs_creation():
    cfg = load_config()
    # basic keys
    assert cfg.get("data_paths.input") is not None
    # dirs exist
    for key in ["data_paths.input", "data_paths.output", "log_paths.log_directory"]:
        p = cfg.get(key)
        if p:
            assert os.path.isdir(p)
