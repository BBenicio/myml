import pandas as pd
from myml.optimization.auto import AutoConfig, AutoML, AutoPipelineChooser, AutoProgressBar, AutoVoting
from myml.optimization.metric import Metric
from myml.optimization.optimizer import OptimizationConfig
from myml.utils import ProblemType


def test_autopipelinechooser(classification_data: pd.DataFrame, classification_target: pd.Series):
    config = AutoConfig(
        ProblemType.classification,
        OptimizationConfig(Metric.f1, evaluations=10, n_jobs=2, seed=24)
    )
    chooser = AutoPipelineChooser(config)
    AutoProgressBar('Auto', chooser)
    
    chooser.numeric_features = classification_data.columns.to_list()
    results = chooser.optimize(classification_data, classification_target)

    assert results.cv_results >= 0.98

def test_automl(classification_data: pd.DataFrame, classification_target: pd.Series):
    config = AutoConfig(
        ProblemType.classification,
        OptimizationConfig(Metric.f1, evaluations=10, n_jobs=2, seed=24)
    )
    chooser = AutoML(config)
    AutoProgressBar('Auto', chooser)
    
    results = chooser.optimize(classification_data, classification_target)

    assert results.cv_results >= 0.97 and results.test_results >= 0.96

def test_autovoting(classification_data: pd.DataFrame, classification_target: pd.Series):
    config = AutoConfig(
        ProblemType.classification,
        OptimizationConfig(Metric.f1, evaluations=10, n_jobs=2, seed=24)
    )
    chooser = AutoVoting(config, voting_count=5, max_candidates=8)
    AutoProgressBar('Auto', chooser)
    
    results = chooser.optimize(classification_data, classification_target)
    voter = results.pipeline.named_steps['votingclassifier']
    assert len(voter.estimators_) == 5
    assert results.cv_results >= 0.97 and results.test_results >= 0.97

def test_auto_regression(regression_data: pd.DataFrame, regression_target: pd.Series):
    config = AutoConfig(
        ProblemType.regression,
        OptimizationConfig(Metric.mse, evaluations=10, n_jobs=2, seed=24)
    )
    chooser = AutoML(config)
    AutoProgressBar('Auto', chooser)
    
    results = chooser.optimize(regression_data, regression_target)

    print(results.cv_results, results.test_results)
    assert results.cv_results >= -3500 and results.test_results >= -3400
