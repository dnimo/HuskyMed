import json
import os
import sys
import tempfile
from types import ModuleType
from unittest.mock import MagicMock

import pandas as pd
import pytest

# --- 构建假的包结构，避免对 flash_attn/transformers 的硬依赖 ---
test_dir = os.path.dirname(os.path.abspath(__file__))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

from test_wrappers import (
    PCWModelWrapper,
    RestrictiveTokensLogitsProcessor,
    PROMPTS,
    TARGET,
    TEXT_BETWEEN_SHOTS,
)

# 构建 model 顶层包
model_pkg = ModuleType('model')
model_pkg.__path__ = []  # 标记为包
sys.modules['model'] = model_pkg

# 构建 model.wrappers 包及其子模块 constants/utils
wrappers_pkg = ModuleType('model.wrappers')
wrappers_pkg.__path__ = []
wrappers_pkg.PCWModelWrapper = PCWModelWrapper
wrappers_pkg.RestrictiveTokensLogitsProcessor = RestrictiveTokensLogitsProcessor
sys.modules['model.wrappers'] = wrappers_pkg

constants_mod = ModuleType('model.wrappers.constants')
constants_mod.PROMPTS = PROMPTS
constants_mod.TARGET = TARGET
constants_mod.TEXT_BETWEEN_SHOTS = TEXT_BETWEEN_SHOTS
sys.modules['model.wrappers.constants'] = constants_mod

utils_mod = ModuleType('model.wrappers.utils')
def filter_extremely_long_samples(df, tokenizer):
    return df
def get_max_n_shots(train_df, test_df, tokenizer, prompt_size):
    return 1
def plot_results_graph(scores_matrix, dataset_name, n_shots_list):
    pass
utils_mod.filter_extremely_long_samples = filter_extremely_long_samples
utils_mod.get_max_n_shots = get_max_n_shots
utils_mod.plot_results_graph = plot_results_graph
sys.modules['model.wrappers.utils'] = utils_mod

# 构建 model.base_loader 模块，提供 load_model_and_tokenizer 替身
base_loader_mod = ModuleType('model.base_loader')
def load_model_and_tokenizer(model_path=None, peft_path=None, torch_dtype=None):
    return MagicMock(), MagicMock(), 'cpu'
base_loader_mod.load_model_and_tokenizer = load_model_and_tokenizer
sys.modules['model.base_loader'] = base_loader_mod

# 导入被测模块
from experiment_manager import ExperimentConfig, ExperimentManager


@pytest.fixture
def minimal_datasets():
    """创建最小训练测试数据集."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 准备训练数据
        train_df = pd.DataFrame({
            "prompts": [
                "请总结以下内容：这是一段很长的文本",
                "请总结以下内容：另一段很长的文本",
            ],
            "target": [
                "文本1总结",
                "文本2总结",
            ]
        })
        train_path = os.path.join(tmpdir, "train.json")
        train_df.to_json(train_path)

        # 准备测试数据
        test_df = pd.DataFrame({
            "prompts": ["请总结以下内容：测试文本"],
            "target": ["测试文本总结"]
        })
        test_path = os.path.join(tmpdir, "test.json")
        test_df.to_json(test_path)

        yield {"train": train_path, "test": test_path, "df": {"train": train_df, "test": test_df}}


def test_load_datasets_basics(minimal_datasets):
    """测试数据加载基本功能."""
    config = ExperimentConfig(
        train_path=minimal_datasets["train"],
        test_path=minimal_datasets["test"],
    )
    manager = ExperimentManager(config)
    manager.load_datasets()

    # 检查数据加载
    assert manager.train_df is not None, "训练数据应该被加载"
    assert manager.test_df is not None, "测试数据应该被加载"
    assert len(manager.train_df) == 2, "训练数据应保持原始行数"
    assert len(manager.test_df) == 1, "测试数据应保持原始行数"
    
    # 检查必要列
    assert "prompts" in manager.train_df.columns, "训练数据应包含prompts列"
    assert "target" in manager.test_df.columns, "测试数据应包含target列"
    
    # 检查标签空间
    assert set(manager.label_space) == {"文本1总结", "文本2总结"}, "标签空间应正确提取"


def test_run_validations(minimal_datasets):
    """测试运行参数验证."""
    config = ExperimentConfig(
        train_path=minimal_datasets["train"],
        test_path=minimal_datasets["test"],
    )
    manager = ExperimentManager(config)

    # 在prepare前调用run应该抛出错误
    with pytest.raises(RuntimeError, match=".*Call initialize_model.*"):
        manager.run(n_shots=[1])

    manager.load_datasets()
    # 注入最小替身以通过 _ensure_ready 检查（后续才验证 n_shots 的入参）
    manager.tokenizer = MagicMock()
    manager.wrapper = PCWModelWrapper(model=MagicMock(), tokenizer=MagicMock(), device='cpu')
    # 空的n_shots列表应该抛出错误
    with pytest.raises(ValueError, match="n_shots list must not be empty"):
        manager.run(n_shots=[])


@pytest.mark.integration
def test_prepare_with_test_doubles(minimal_datasets):
    """在不初始化真实模型的情况下，使用替身通过 prepare 阶段。"""
    config = ExperimentConfig(
        train_path=minimal_datasets['train'],
        test_path=minimal_datasets['test'],
        context_window_size=512,
        progress_bar=False,
    )
    manager = ExperimentManager(config)
    manager.load_datasets()

    # 手动注入 tokenizer 与 wrapper 以通过 _ensure_ready 检查
    manager.tokenizer = MagicMock()
    manager.wrapper = PCWModelWrapper(model=MagicMock(), tokenizer=MagicMock(), device='cpu')

    # prepare 依赖 utils 中的函数，已在上方提供替身
    manager.prepare()
    assert manager.max_n_shots == 1