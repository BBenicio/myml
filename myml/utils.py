from enum import IntEnum
import numpy as np
import pandas as pd


class DataType(IntEnum):
    numeric = 1
    categorical = 2


class ProblemType(IntEnum):
    classification = 1
    regression = 2


Features = np.ndarray | pd.DataFrame
Target = np.ndarray | pd.Series
