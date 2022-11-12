""" Utilities for on the rest of the package. """

from enum import IntEnum
from typing import List
import numpy as np
import pandas as pd


class DataType(IntEnum):
    """ Type of a feature. """

    numeric = 1
    """ Represent a numeric feature. """

    categorical = 2
    """ Represent a categorical feature. """


class ProblemType(IntEnum):
    """ Kind of problem to solve. """

    classification = 1
    """ Classification problems, to predict discrete class labels. """

    regression = 2
    """ Regression problems, to predict continuous numbers. """


Features = np.ndarray | pd.DataFrame
""" Type for the input feature matrix. """

Target = np.ndarray | pd.Series
""" Type for the target values vector. """


def get_features_labels(X: Features) -> List:
    """
    Get a list of feature names.

    Retrieve the column labels for the input feature matrix.

    Parameters
    ----------
    X : Features
        Feature matrix from which to retrieve the names.

    Returns
    -------
    list
        List of the feature labels.
    """
    if type(X) is np.ndarray:
        return np.arange(X.shape[1]).tolist()
    elif type(X) is pd.DataFrame:
        return X.columns.to_list()


def filter_by_types(X: Features, types: List[str]) -> pd.DataFrame:
    """
    Get only features from a select data type from the feature matrix.

    Filter the feature matrix columns, keeping only columns where the data
    type matches one of the given types.

    Parameters
    ----------
    X : Features
        Feature matrix to filter.
    types : list of str
        Data types to keep, should be one of the pandas dtypes.

    Returns
    -------
    pandas.DataFrame
        Filtered feature matrix containing only columns of the given types.

    See Also
    --------
    pandas.DataFrame.select_dtypes : Select columns by type.
    """
    df: pd.DataFrame = None
    if type(X) is np.ndarray:
        df = pd.DataFrame(X)
    elif type(X) is pd.DataFrame:
        df = X

    return df.select_dtypes(include=types)
