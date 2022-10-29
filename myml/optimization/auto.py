from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import myml
from myml import models
from myml.optimization.metric import Metric
from myml.optimization.optimizer import OptimizationConfig, OptimizerProgressBar, PipelineChooser
from myml.optimization.search import PipelineSearchSpace, Preprocessor
from myml.utils import DataType, Features, ProblemType, Target
from myml.preprocessors import categorical_preprocessors, numeric_preprocessors
from sklearn.base import TransformerMixin
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import get_scorer
import numpy as np
import pandas as pd


class AutoConfig(NamedTuple):
    problem_type: ProblemType
    optimization_config: OptimizationConfig


class AutoPipelineChooser:
    def __init__(self, config: AutoConfig) -> None:
        self.config = config

        self.pipeline_chooser = PipelineChooser(config.optimization_config)
        self.data_types: Dict[DataType, List[Any]] = {}
    
    @property
    def categorical_features(self) -> List[Any]:
        return self.data_types[DataType.categorical] if DataType.categorical in self.data_types else []
    
    @categorical_features.setter
    def categorical_features(self, features: List[Any]):
        self.data_types[DataType.categorical] = features
    
    @property
    def numeric_features(self) -> List[Any]:
        return self.data_types[DataType.numeric] if DataType.numeric in self.data_types else []
    
    @numeric_features.setter
    def numeric_features(self, features: List[Any]):
        self.data_types[DataType.numeric] = features
    
    def _set_model_chooser_search_space(self):
        self.pipeline_chooser.model_chooser.search_space = models.estimators[self.config.problem_type]
    
    def _combine_into_pipelines(self, features: List, preprocessors: List[TransformerMixin], pipelines: List[List[Preprocessor]]) -> List[List[Preprocessor]]:
        new_pipelines = []
        if len(features) > 0:
            options = [(preprocessor, features) for preprocessor in preprocessors]
            for pipe in pipelines:
                for op in options:
                    new_pipelines.append(pipe + [op])
        else:
            new_pipelines = pipelines

        return new_pipelines

    def _assemble_search_space(self, X: Features):
        self._set_model_chooser_search_space()
        
        features_have_nan = np.any(np.isnan(X))
        pipelines = [[SimpleImputer()]] if features_have_nan else [[]]
        pipelines = self._combine_into_pipelines(self.numeric_features, numeric_preprocessors, pipelines)
        pipelines = self._combine_into_pipelines(self.categorical_features, categorical_preprocessors, pipelines)
        self.pipeline_chooser.search_space = PipelineSearchSpace(**{f'pipeline_{i}': pipeline for i, pipeline in enumerate(pipelines)})
        
    def optimize(self, X: Features, y: Target) -> Tuple[Pipeline, float]:
        self._assemble_search_space(X)
        results = self.pipeline_chooser.optimize(X, y)
        return make_pipeline(results.column_transformer, results.estimator), results.evaluation


class AutoML(AutoPipelineChooser):
    def __init__(self, config: AutoConfig, test_size: float = 0.3) -> None:
        super().__init__(config)
        self.test_size = test_size
    
    def _setup_features(self, X: Features, override_numeric_features: Optional[List[str]] = None, override_categorical_features: Optional[List[str]] = None) -> None:
        if override_numeric_features is None:
            self.numeric_features = X.select_dtypes(include=['number']).columns.to_list()
        else:
            self.numeric_features = override_numeric_features

        if override_categorical_features is None:
            self.categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.to_list()
        else:
            self.categorical_features = override_categorical_features
        
    def _split_train_test(self, X: Features, y: Target) -> Tuple[Features, Features, Target, Target]:
        if self.config.problem_type == ProblemType.classification:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.config.optimization_config.seed, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.config.optimization_config.seed)
        return X_train, X_test, y_train, y_test


    def optimize(self, X: Features, y: Target, override_numeric_features: Optional[List[str]] = None, override_categorical_features: Optional[List[str]] = None) -> Tuple[Pipeline, float, float]:
        self._setup_features(X, override_numeric_features, override_categorical_features)
        X_train, X_test, y_train, y_test = self._split_train_test(X, y)

        pipeline, result = super().optimize(X_train, y_train)

        pipeline.fit(X_train, y_train)
        scorer = get_scorer(self.config.optimization_config.metric.value.sklearn_name)
        test_score = scorer(pipeline, X_test, y_test)
        
        return pipeline, result, test_score


class AutoProgressBar:
    def __init__(self, name: str, auto: AutoPipelineChooser) -> None:
        self.name = name
        self.pipeline = OptimizerProgressBar(f'{name} PipelineChooser', auto.pipeline_chooser)
        self.model = OptimizerProgressBar(f'{name} ModelChooser', auto.pipeline_chooser.model_chooser)
        self.hyperparameter = OptimizerProgressBar(f'{name} HyperparameterOptimizer', auto.pipeline_chooser.model_chooser.optimizer)
