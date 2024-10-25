import logging
import random
from typing import List, Dict

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from ParallelContextsWindows.constants import PROMPTS, N_TOKENS, TEXT_BETWEEN_SHOTS
from ParallelContextsWindows.pcw_wrapper import PCWModelWrapper
from ParallelContextsWindows.logits_processor import RestrictiveTokensLogitsProcessor
from ParallelContextsWindows.utils import get_max_n_shots, filter_extremely_long_samples, plot_results_graph
from tools.Model.common.model_loaders import load_pcw_wrapper

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

class ExperimentManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.context_window_size = None
        self.right_indentation = None
        self.max_n_shots = None
        self.n_shots = None
        self.train_df = None
        self.test_df = None
        self.results = None
