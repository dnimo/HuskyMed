"""Global test configuration and safe stubs to avoid heavy deps during import.

要点：
- 不再 stub 已安装的 `datasets`，以免破坏其 API（之前导致 concatenate_datasets 缺失）。
- 提供一个轻量的 `HuskyMed` 包占位，阻止 pytest 在收集时执行项目根 __init__.py。
- 保留对 `flash_attn` 的占位，避免缺失 CUDA 等环境时报错。
"""

import sys
import importlib.machinery
from types import ModuleType
from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config):
    """避免收集并导入项目根的 __init__.py 造成重依赖加载。"""
    try:
        p = Path(str(collection_path))
    except Exception:
        return False
    # 忽略项目根 __init__.py（绝对与相对兜底）
    if p.name == "__init__.py" and p.parent.name == "HuskyMed":
        return True
    if str(p).endswith("/HuskyMed/__init__.py"):
        return True
    return False


# 注册一个轻量的 HuskyMed 包占位，避免 pytest 导入真实 __init__.py
if 'HuskyMed' not in sys.modules:
    husky_pkg = ModuleType('HuskyMed')
    # 作为包占位符
    husky_pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
    husky_pkg.__spec__ = importlib.machinery.ModuleSpec('HuskyMed', loader=None, is_package=True)
    sys.modules['HuskyMed'] = husky_pkg


# Stub out heavy optional dependencies that may be imported at package init time
# 注意：不要 stub `datasets`，用户已安装；之前的 stub 会导致缺少 concatenate_datasets 等符号。
if 'flash_attn' not in sys.modules:
    _m = ModuleType('flash_attn')
    _m.__spec__ = importlib.machinery.ModuleSpec('flash_attn', loader=None)
    sys.modules['flash_attn'] = _m
