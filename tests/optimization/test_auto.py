import pandas as pd
from myml.optimization.auto import AutoML, AutoPipelineChooser
from myml.optimization.metric import Metric
from myml.utils import ProblemType

def test_autopipelinechooser(data: pd.DataFrame, target: pd.Series):
    chooser = AutoPipelineChooser(ProblemType.classification, Metric.f1, evaluations=10, n_jobs=-1, seed=24)
    chooser.numeric_features = data.columns.to_list()
    _, result = chooser.optimize(data, target)
    assert result >= 0.96

def test_automl(data: pd.DataFrame, target: pd.Series):
    chooser = AutoML(ProblemType.classification, Metric.f1, evaluations=10, n_jobs=-1, seed=24)
    _, result, test_result = chooser.optimize(data, target, voting=5)

    assert result >= 0.97 and test_result >= 0.96