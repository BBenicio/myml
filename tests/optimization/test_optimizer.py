import pandas as pd
from myml.optimization.optimizer import OptimizationConfig, PipelineChooser, ModelChooser, HyperparameterOptimizer, OptimizerProgressBar, CashOptimizer
from myml.optimization.metric import Metric
from myml.optimization.search import HyperparameterSearchSpace, PipelineSearchSpace
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from skopt.space import Real, Integer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, FunctionTransformer


def test_hyperparameteroptimizer(classification_data: pd.DataFrame, classification_target: pd.Series):
    est = GradientBoostingClassifier(n_estimators=50, random_state=0)
    config = OptimizationConfig(Metric.f1, evaluations=10, cv=5, n_jobs=2, seed=0)
    optimizer = HyperparameterOptimizer(est, config)
    optimizer.search_space = HyperparameterSearchSpace(
        max_depth = Integer(1, 5),
        learning_rate = Real(1e-5, 1e0, 'log-uniform'),
        min_samples_split = Integer(2, 100),
        min_samples_leaf = Integer(1, 100)
    )
    results = optimizer.optimize(classification_data, classification_target)
    
    assert results.hyperparameters['max_depth'] == 3
    assert results.evaluation >= 0.8

def test_modelchooser(classification_data: pd.DataFrame, classification_target: pd.Series):
    config = OptimizationConfig(Metric.f1, evaluations=10, cv=5, n_jobs=2, seed=0)
    chooser = ModelChooser(config)
    
    chooser.search_space[GradientBoostingClassifier(n_estimators=50, random_state=0)] = HyperparameterSearchSpace(
        max_depth = Integer(1, 5),
        learning_rate = Real(1e-5, 1e0, 'log-uniform'),
        min_samples_split = Integer(2, 100),
        min_samples_leaf = Integer(1, 100)
    )
    
    chooser.search_space[RandomForestClassifier(n_estimators=50, random_state=0)] = HyperparameterSearchSpace(
        max_depth = Integer(1, 5),
        min_samples_split = Integer(2, 100),
        min_samples_leaf = Integer(1, 100)
    )

    results = chooser.optimize(classification_data, classification_target)

    assert isinstance(results.estimator, GradientBoostingClassifier)
    assert results.hyperparameters['max_depth'] == 3
    assert results.evaluation >= 0.8

def test_pipelinechooser(classification_data: pd.DataFrame, classification_target: pd.Series):
    config = OptimizationConfig(Metric.f1, evaluations=10, cv=5, n_jobs=2, seed=0)

    chooser = PipelineChooser(config)
    
    chooser.search_space = PipelineSearchSpace(
        identity = [ (FunctionTransformer(), [0,1,2,3,4,5,6,7]) ],
        standard = [ (StandardScaler(), [0,1,2,3,4,5,6,7]) ],
        min_max_max_abs = [ (MinMaxScaler(), [1,2,4,6,7]), (MaxAbsScaler(), [0,3,5]) ]
    )
    
    chooser.model_chooser.search_space[GradientBoostingClassifier(n_estimators=50, random_state=0)] = HyperparameterSearchSpace(
        max_depth = Integer(1, 5),
        learning_rate = Real(1e-5, 1e0, 'log-uniform'),
        min_samples_split = Integer(2, 100),
        min_samples_leaf = Integer(1, 100)
    )

    chooser.model_chooser.search_space[RandomForestClassifier(n_estimators=50, random_state=0)] = HyperparameterSearchSpace(
        max_depth = Integer(1, 5),
        min_samples_split = Integer(2, 100),
        min_samples_leaf = Integer(1, 100)
    )

    results = chooser.optimize(classification_data, classification_target)
    assert isinstance(results.column_transformer.transformers[0][1], MinMaxScaler)
    assert isinstance(results.estimator, GradientBoostingClassifier)
    assert results.hyperparameters['max_depth'] == 2
    assert results.evaluation >= 0.9

def test_cashoptimizer(classification_data: pd.DataFrame, classification_target: pd.Series):
    config = OptimizationConfig(Metric.f1, evaluations=20, cv=5, n_jobs=2, seed=0)
    optimizer = CashOptimizer(config)
    
    optimizer.search_space[GradientBoostingClassifier(n_estimators=50, random_state=0)] = HyperparameterSearchSpace(
        max_depth = Integer(1, 5),
        learning_rate = Real(1e-5, 1e0, 'log-uniform'),
        min_samples_split = Integer(2, 100),
        min_samples_leaf = Integer(1, 100)
    )
    
    optimizer.search_space[RandomForestClassifier(n_estimators=50, random_state=0)] = HyperparameterSearchSpace(
        max_depth = Integer(1, 5),
        min_samples_split = Integer(2, 100),
        min_samples_leaf = Integer(1, 100)
    )

    results = optimizer.optimize(classification_data, classification_target)

    assert isinstance(results.estimator, GradientBoostingClassifier)
    assert results.hyperparameters['max_depth'] == 2
    assert results.evaluation >= 0.9
