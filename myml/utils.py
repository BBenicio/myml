from enum import IntEnum
from typing import List
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

def get_features_labels(X: Features) -> List:
    if type(X) is np.ndarray:
        return np.arange(X.shape[1]).tolist()
    elif type(X) is pd.DataFrame:
        return X.columns.to_list()

def filter_by_types(X: Features, types: List[str]) -> pd.DataFrame:
    df: pd.DataFrame = None
    if type(X) is np.ndarray:
        df = pd.DataFrame(X)
    elif type(X) is pd.DataFrame:
        df = X

    return df.select_dtypes(include=types)
