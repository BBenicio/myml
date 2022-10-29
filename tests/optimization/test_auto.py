import pandas as pd
from myml.optimization.auto import AutoConfig, AutoML, AutoPipelineChooser, AutoProgressBar
from myml.optimization.metric import Metric
from myml.optimization.optimizer import OptimizationConfig
from myml.utils import ProblemType

def test_autopipelinechooser(data: pd.DataFrame, target: pd.Series):
    config = AutoConfig(
        ProblemType.classification,
        OptimizationConfig(Metric.f1, evaluations=10, n_jobs=2, seed=24)
    )
    chooser = AutoPipelineChooser(config)
    AutoProgressBar('Auto', chooser)
    
    chooser.numeric_features = data.columns.to_list()
    _, result = chooser.optimize(data, target)
    print(result)
    assert result >= 0.98

def test_automl(data: pd.DataFrame, target: pd.Series):
    config = AutoConfig(
        ProblemType.classification,
        OptimizationConfig(Metric.f1, evaluations=10, n_jobs=2, seed=24)
    )
    chooser = AutoML(config)
    AutoProgressBar('Auto', chooser)
    
    _, result, test_result = chooser.optimize(data, target)

    print(result, test_result)
    assert result >= 0.97 and test_result >= 0.97