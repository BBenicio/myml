from enum import IntEnum
import numpy as np
import pandas as pd
import tqdm
from optimization.optimizer import Optimizer
from typing import Any, Dict


class DataType(IntEnum):
    numeric = 1
    categorical = 2
    # ordinal = 3
    # textual = 4


class ProblemType(IntEnum):
    classification = 1
    regression = 2


Features = np.ndarray | pd.DataFrame
Target = np.ndarray | pd.Series


class OptimizerProgressBar:
    def __init__(self, name: str, optimizer: Optimizer) -> None:
        self.name = name
        optimizer.progress_callback = self._callback
        self.pbar: tqdm.tqdm = None
        self.i = 0
    
    def _callback(self, data: Dict[str, Any], total: int) -> None:
        self.i += 1
        if self.pbar is None:
            self.pbar = tqdm.tqdm(total=total, desc=self.name)
        
        if data is not None:
            self.pbar.set_postfix(data)

        self.pbar.update(1)
        
        if self.i == total:
            self.i = 0
            self.pbar.close()
            self.pbar = None
