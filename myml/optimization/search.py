"""
Search spaces for the optimizers.

For optimization problems we have a space to explore when searching for
the best solution. This module provides classes for each of the
optimization problems on the package.
"""

from typing import Iterator, List, NamedTuple
from skopt.space import Dimension
from sklearn.base import BaseEstimator, TransformerMixin


class SearchSpace(dict):
    """ Generic search space, a dictionary. """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class HyperparameterSearchSpace(SearchSpace):
    """
    Search space for hyperparameter optimization.

    Parameters
    ----------
    **kwargs
        Hyperparameters to search on, each should be a `skopt.Dimension`.
    """

    def __init__(self, **kwargs: Dimension) -> None:
        for k in kwargs:
            kwargs[k].name = k
        super().__init__(**kwargs)

    def __getitem__(self, key: str) -> Dimension:
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Dimension) -> None:
        value.name = key
        super().__setitem__(key, value)

    def __iter__(self) -> Iterator[Dimension]:
        yield from self.values()


class ModelSearchSpace(SearchSpace):
    """ Search space for algorighm selection and hyperparameter optimization. """

    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, key: BaseEstimator) -> HyperparameterSearchSpace:
        return super().__getitem__(key)

    def __setitem__(self, key: BaseEstimator, value: HyperparameterSearchSpace) -> None:
        super().__setitem__(key, value)

    def __iter__(self) -> Iterator[BaseEstimator]:
        yield from self.keys()


class Preprocessor(NamedTuple):
    """
    Feature preprocessor.

    Parameters
    ----------
    transformer : TransformerMixin
        Scikit-learn compatible transformer for the features.
    columns : list of str
        List of features to which apply the transformer.

    Attributes
    ----------
    transformer : TransformerMixin
        Scikit-learn compatible transformer for the features.
    columns : list of str
        List of features to which apply the transformer.
    """
    transformer: TransformerMixin
    columns: List[str]


class PipelineSearchSpace(SearchSpace):
    """ Search space for preprocessing pipeline selection. """

    def __init__(self, **kwargs: List[Preprocessor]) -> None:
        super().__init__(**kwargs)

    def __getitem__(self, key: str) -> List[Preprocessor]:
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: List[Preprocessor]) -> None:
        return super().__setitem__(key, value)

    def __iter__(self) -> Iterator[List[Preprocessor]]:
        yield from self.values()
