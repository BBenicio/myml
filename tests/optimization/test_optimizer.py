import pandas as pd
from myml.optimization.optimizer import PipelineChooser, ModelChooser, HyperparameterOptimizer
from myml.optimization.metric import Metric
from myml.utils import ProblemType
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from skopt.space import Real, Integer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, FunctionTransformer


def test_hyperparameteroptimizer(data: pd.DataFrame, target: pd.Series):
    est = GradientBoostingClassifier(n_estimators=50, random_state=0)
    optimizer = HyperparameterOptimizer(est, Metric.f1, evaluations=10, cv=5, n_jobs=-1, seed=0)
    optimizer.search_space = {
        'max_depth': Integer(1, 5),
        'learning_rate': Real(1e-5, 1e0, 'log-uniform'),
        'min_samples_split': Integer(2, 100),
        'min_samples_leaf': Integer(1, 100)
    }
    params, result = optimizer.optimize(data, target)
    
    assert params['max_depth'] == 3
    assert result >= 0.8

def test_modelchooser(data: pd.DataFrame, target: pd.Series):
    chooser = ModelChooser(Metric.f1, evaluations=10, cv=5, n_jobs=-1, seed=0)
    chooser.search_space = {
        GradientBoostingClassifier(n_estimators=50, random_state=0): {
            'max_depth': Integer(1, 5),
            'learning_rate': Real(1e-5, 1e0, 'log-uniform'),
            'min_samples_split': Integer(2, 100),
            'min_samples_leaf': Integer(1, 100)
        },
        RandomForestClassifier(n_estimators=50, random_state=0): {
            'max_depth': Integer(1, 5),
            'min_samples_split': Integer(2, 100),
            'min_samples_leaf': Integer(1, 100)
        }
    }

    est, params, result = chooser.optimize(data, target)

    assert isinstance(est, GradientBoostingClassifier)
    assert params['max_depth'] == 3
    assert result >= 0.8

def test_pipelinechooser(data: pd.DataFrame, target: pd.Series):
    chooser = PipelineChooser(Metric.f1, evaluations=10, cv=5, n_jobs=-1, seed=0)
    chooser.search_space = {
        GradientBoostingClassifier(n_estimators=50, random_state=0): {
            'max_depth': Integer(1, 5),
            'learning_rate': Real(1e-5, 1e0, 'log-uniform'),
            'min_samples_split': Integer(2, 100),
            'min_samples_leaf': Integer(1, 100)
        },
        RandomForestClassifier(n_estimators=50, random_state=0): {
            'max_depth': Integer(1, 5),
            'min_samples_split': Integer(2, 100),
            'min_samples_leaf': Integer(1, 100)
        }
    }

    chooser.pipeline_search_space = [
        [ (FunctionTransformer(), [0,1,2,3,4,5,6,7]) ],
        [ (StandardScaler(), [0,1,2,3,4,5,6,7]) ],
        [ (MinMaxScaler(), [1,2,4,6,7]), (MaxAbsScaler(), [0,3,5]) ],
    ]

    ct, est, params, result = chooser.optimize(data, target)
    assert isinstance(ct.transformers[0][1], MinMaxScaler)
    assert isinstance(est, GradientBoostingClassifier)
    assert params['max_depth'] == 2
    assert result >= 0.9