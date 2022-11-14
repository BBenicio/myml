import pytest
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

@pytest.fixture
def classification_dataset() -> pd.DataFrame:
    cancer = load_breast_cancer(as_frame=True)
    return cancer

@pytest.fixture
def classification_data(classification_dataset) -> pd.DataFrame:
    return classification_dataset.data

@pytest.fixture
def classification_target(classification_dataset) -> pd.Series:
    return classification_dataset.target

@pytest.fixture
def regression_dataset() -> pd.DataFrame:
    diabetes = load_diabetes(as_frame=True)
    return diabetes

@pytest.fixture
def regression_data(regression_dataset) -> pd.DataFrame:
    return regression_dataset.data

@pytest.fixture
def regression_target(regression_dataset) -> pd.Series:
    return regression_dataset.target
