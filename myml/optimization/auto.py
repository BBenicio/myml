from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple
from myml import models
from myml.optimization.optimizer import OptimizationConfig, OptimizerProgressBar, PipelineChooser
from myml.optimization.search import PipelineSearchSpace, Preprocessor
from myml.utils import DataType, Features, ProblemType, Target, filter_by_types, get_features_labels
from myml.preprocessors import categorical_preprocessors, numeric_preprocessors, imputers
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer
import numpy as np


class AutoConfig(NamedTuple):
    problem_type: ProblemType
    optimization_config: OptimizationConfig


class AutoResults(NamedTuple):
    pipeline: Pipeline
    cv_results: float
    test_results: Optional[float] = None


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
    
    def _get_starting_pipeline(self, X: Features) -> List[List[Preprocessor]]:
        features_have_nan = np.any(np.isnan(X))
        if features_have_nan:
            return [[Preprocessor(imputer, get_features_labels(X))] for imputer in imputers]
        else:
            return [[]]
    
    def _generate_options(self, features: List, transformers: List[TransformerMixin]) -> Iterator[Preprocessor]:
        return (Preprocessor(transformer, features) for transformer in transformers)

    def _add_options_to_pipelines(self, pipelines: List[List[Preprocessor]], options: List[Preprocessor]) -> List[List[Preprocessor]]:
        new = []
        for pipeline in pipelines:
            for op in options:
                new.append(pipeline + [op])

        return new
    
    def _combine_with_pipelines(self, features: List, transformers: List[TransformerMixin], pipelines: List[List[Preprocessor]]) -> List[List[Preprocessor]]:
        if len(features) == 0:
            return pipelines

        options = self._generate_options(features, transformers)
        new_pipelines = self._add_options_to_pipelines(pipelines, options)
        
        return new_pipelines
    
    def _convert_pipelines(self, pipelines: List[List[Preprocessor]]) -> PipelineSearchSpace:
        converted_pipelines = {f'pipeline{i}': pipeline for i, pipeline in enumerate(pipelines)}
        return PipelineSearchSpace(**converted_pipelines)

    def _assemble_search_space(self, X: Features):
        self._set_model_chooser_search_space()
        
        pipelines = self._get_starting_pipeline(X)
        pipelines = self._combine_with_pipelines(self.numeric_features, numeric_preprocessors, pipelines)
        pipelines = self._combine_with_pipelines(self.categorical_features, categorical_preprocessors, pipelines)
        self.pipeline_chooser.search_space = self._convert_pipelines(pipelines)
        
    def optimize(self, X: Features, y: Target) -> AutoResults:
        self._assemble_search_space(X)
        results = self.pipeline_chooser.optimize(X, y)
        return AutoResults(make_pipeline(results.column_transformer, results.estimator), results.evaluation)


class AutoML(AutoPipelineChooser):
    def __init__(self, config: AutoConfig, test_size: float = 0.3) -> None:
        super().__init__(config)
        self.test_size = test_size
    
    def _get_default_or_override(self, X: Features, types: List[str], override: Optional[List[str]] = None) -> List:
        if override is None:
            return get_features_labels(filter_by_types(X, ['number']))
        else:
            return override
    
    def _setup_features(self, X: Features, override_numeric_features: Optional[List[str]] = None, override_categorical_features: Optional[List[str]] = None) -> None:
        self.numeric_features = self._get_default_or_override(X, ['number'], override_numeric_features)
        self.categorical_features = self._get_default_or_override(X, ['object', 'category', 'bool'], override_categorical_features)
        
    def _split_train_test(self, X: Features, y: Target) -> Tuple[Features, Features, Target, Target]:
        if self.config.problem_type == ProblemType.classification:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.config.optimization_config.seed, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.config.optimization_config.seed)
        return X_train, X_test, y_train, y_test


    def optimize(self, X: Features, y: Target, override_numeric_features: Optional[List[str]] = None, override_categorical_features: Optional[List[str]] = None) -> AutoResults:
        self._setup_features(X, override_numeric_features, override_categorical_features)
        X_train, X_test, y_train, y_test = self._split_train_test(X, y)

        results = super().optimize(X_train, y_train)

        results.pipeline.fit(X_train, y_train)
        scorer = get_scorer(self.config.optimization_config.metric.value.sklearn_name)
        test_score = scorer(results.pipeline, X_test, y_test)
        
        return AutoResults(results.pipeline, results.cv_results, test_score)


class AutoProgressBar:
    def __init__(self, name: str, auto: AutoPipelineChooser) -> None:
        self.name = name
        self.pipeline = OptimizerProgressBar(f'{name} PipelineChooser', auto.pipeline_chooser)
        self.model = OptimizerProgressBar(f'{name} ModelChooser', auto.pipeline_chooser.model_chooser)
        self.hyperparameter = OptimizerProgressBar(f'{name} HyperparameterOptimizer', auto.pipeline_chooser.model_chooser.optimizer)
