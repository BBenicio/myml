import pytest
import pandas as pd
from sklearn.datasets import load_breast_cancer

@pytest.fixture
def dataset() -> pd.DataFrame:
    cancer = load_breast_cancer(as_frame=True)
    return cancer

@pytest.fixture
def data(dataset) -> pd.DataFrame:
    return dataset.data

@pytest.fixture
def target(dataset) -> pd.Series:
    return dataset.target
