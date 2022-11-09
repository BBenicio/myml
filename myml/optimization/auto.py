"""
Get machine learning models for data without setting up search spaces.

This module provides an interface for automatic machine learning, so that
it is possible to use the optimization techniques to get good machine
learning models without the need to manually set search spaces.
"""

import numpy as np
import warnings
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple
from myml import models
from myml.optimization.metric import is_better, sort_by_metric
from myml.optimization.optimizer import OptimizationConfig, OptimizationResults, OptimizerProgressBar, PipelineChooser, CashOptimizer
from myml.optimization.search import PipelineSearchSpace, Preprocessor
from myml.utils import DataType, Features, ProblemType, Target, filter_by_types, get_features_labels
from myml.preprocessors import categorical_preprocessors, numeric_preprocessors, imputers
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import get_scorer
from sklearn.ensemble import VotingClassifier, VotingRegressor


class AutoConfig(NamedTuple):
    """
    Configurations for the automatic optimization.

    Parameters
    ----------
    problem_type : ProblemType
        The kind of machine learning problem for which to build models.
    optimization_config : OptimizationConfig
        Configuration for the underlying optimization process.

    Attributes
    ----------
    problem_type : ProblemType
        The kind of machine learning problem for which to build models.
    optimization_config : OptimizationConfig
        Configuration for the underlying optimization process.
    """
    problem_type: ProblemType
    optimization_config: OptimizationConfig


class AutoResults(NamedTuple):
    """
    Results from the automatic optimization process.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline built for the data
    cv_results : float
        Results from the cross-validation on the training data.
    test_results : float, optional
        Results from the testing set.

    Attributes
    ----------
    pipeline : Pipeline
        The pipeline built for the data
    cv_results : float
        Results from the cross-validation on the training data.
    test_results : float, optional
        Results from the testing set.
    """
    pipeline: Pipeline
    cv_results: float
    test_results: Optional[float] = None


class AutoPipelineChooser:
    """
    Auto optimizer for choosing data pipelines and algorithms.

    Given the training data, find the data pipeline, algorithm and
    hyperparameter combination that works best for it.

    Parameters
    ----------
    config : AutoConfig
        Configuration for the automatic optimizer.

    Attributes
    ----------
    config : AutoConfig
        Configuration for the automatic optimizer.
    pipeline_chooser : PipelineChooser
        Underlying pipeline optimizer that searches the search space.
    categorical_features : list
        List of categorical features to pass to the data pipeline.
    numeric_features : list
        List of numeric features to pass to the data pipeline.
    """

    def __init__(self, config: AutoConfig) -> None:
        self.config = config

        self.pipeline_chooser = PipelineChooser(
            config.optimization_config, CashOptimizer(config.optimization_config))
        self._data_types: Dict[DataType, List[Any]] = {}

    @property
    def categorical_features(self) -> List[Any]:
        """ List of categorical features to pass to the data pipeline. """
        return self._data_types[DataType.categorical] if DataType.categorical in self._data_types else []

    @categorical_features.setter
    def categorical_features(self, features: List[Any]):
        self._data_types[DataType.categorical] = features

    @property
    def numeric_features(self) -> List[Any]:
        """ List of numeric features to pass to the data pipeline. """
        return self._data_types[DataType.numeric] if DataType.numeric in self._data_types else []

    @numeric_features.setter
    def numeric_features(self, features: List[Any]):
        self._data_types[DataType.numeric] = features

    def _set_model_chooser_search_space(self):
        """ Set the algorithm search space before optimization. """
        self.pipeline_chooser.model_chooser.search_space = models.estimators[
            self.config.problem_type]

    def _get_starting_pipeline(self, X: Features) -> List[List[Preprocessor]]:
        """
        Get the base of the data pipeline search space.

        Initialize the data pipeline either empty or with and imputing
        step if there are missing values.

        Parameters
        ----------
        X : Features
            Features to use to check for missing values.
        Returns
        -------
        list of list of Preprocessor
            Initial data pipeline to search.
        """
        features_have_nan = np.any(np.isnan(X))
        if features_have_nan:
            return [[Preprocessor(imputer, get_features_labels(X))] for imputer in imputers]
        else:
            return [[]]

    def _generate_options(self, features: List, transformers: List[TransformerMixin]) -> Iterator[Preprocessor]:
        """
        Combine the transformers and features to add to the search space.

        Parameters
        ----------
        features : list
            List of feature labels to apply the transformers.
        transformers : list of sklearn.base.TransformerMixin
            List of transformers that can be applied to the features.

        Returns
        -------
        Iterator of Preprocessor
            Preprocessors associating each transformer with the features.
        """
        return (Preprocessor(transformer, features) for transformer in transformers)

    def _add_options_to_pipelines(self, pipelines: List[List[Preprocessor]], options: List[Preprocessor]) -> List[List[Preprocessor]]:
        """
        Add the options to the pipeline steps search space.

        Parameters
        ----------
        pipelines : list of list of Preprocessor
            Initial data pipeline to increment.
        options : list of Preprocessor
            Preprocessing options to add on the pipeline.

        Returns
        -------
        list of list of Preprocessor
            New data pipeline to search.
        """
        new = []
        for pipeline in pipelines:
            for op in options:
                new.append(pipeline + [op])

        return new

    def _combine_with_pipelines(self, features: List, transformers: List[TransformerMixin], pipelines: List[List[Preprocessor]]) -> List[List[Preprocessor]]:
        """
        Put the transformers into the pipeline search space.

        Parameters
        ----------
        features : list
            List of feature labels to apply the transformers.
        transformers : list of sklearn.base.TransformerMixin
            List of transformers that can be applied to the features.
        pipelines : list of list of Preprocessor
            Initial data pipeline to increment.

        Returns
        -------
        list of list of Preprocessor
            New data pipeline to search.
        """

        if len(features) == 0:
            return pipelines

        options = self._generate_options(features, transformers)
        new_pipelines = self._add_options_to_pipelines(pipelines, options)

        return new_pipelines

    def _convert_pipelines(self, pipelines: List[List[Preprocessor]]) -> PipelineSearchSpace:
        """
        Convert the data pipeline assembled into the `PipelineSearchSpace`.

        Parameters
        ----------
        pipelines : list of list of Preprocessor
            Data pipeline options to be searched.

        Returns
        -------
        PipelineSearchSpace
            Formalized search space for the data pipeline options.
        """
        converted_pipelines = {
            f'pipeline{i}': pipeline for i, pipeline in enumerate(pipelines)}
        return PipelineSearchSpace(**converted_pipelines)

    def _assemble_search_space(self, X: Features):
        """
        Setup the search spaces for the optimization to run.

        Parameters
        ----------
        X : Features
            Features to use to check if special steps are needed.
        """
        self._set_model_chooser_search_space()

        pipelines = self._get_starting_pipeline(X)
        pipelines = self._combine_with_pipelines(
            self.numeric_features, numeric_preprocessors, pipelines)
        pipelines = self._combine_with_pipelines(
            self.categorical_features, categorical_preprocessors, pipelines)
        self.pipeline_chooser.search_space = self._convert_pipelines(pipelines)

    def optimize(self, X: Features, y: Target) -> AutoResults:
        """
        Optimize the data pipeline and algorithm for the given data.

        Find the data pipeline, algorithm and hyperparameters that get the
        best performing cross-validation metrics.

        Parameters
        ----------
        X : Features
            Features to use for model training
        y : Target
            Target values for model training

        Returns
        -------
        AutoResults
            Results from the automatic optimization.
        """
        self._assemble_search_space(X)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            results = self.pipeline_chooser.optimize(X, y)

        return AutoResults(make_pipeline(results.column_transformer, results.estimator), results.evaluation)


class AutoML(AutoPipelineChooser):
    def __init__(self, config: AutoConfig, test_size: float = 0.3) -> None:
        super().__init__(config)
        self.test_size = test_size

    def _get_default_or_override(self, X: Features, types: List[str], override: Optional[List[str]] = None) -> List:
        if override is None:
            return get_features_labels(filter_by_types(X, types))
        else:
            return override

    def _setup_features(self, X: Features, override_numeric_features: Optional[List[str]] = None, override_categorical_features: Optional[List[str]] = None) -> None:
        self.numeric_features = self._get_default_or_override(
            X, ['number'], override_numeric_features)
        self.categorical_features = self._get_default_or_override(
            X, ['object', 'category', 'bool'], override_categorical_features)

    def _split_train_test(self, X: Features, y: Target) -> Tuple[Features, Features, Target, Target]:
        if self.config.problem_type == ProblemType.classification:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.config.optimization_config.seed, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.config.optimization_config.seed)
        return X_train, X_test, y_train, y_test

    def _fit_and_test(self, estimator: Pipeline, X_train: Features, y_train: Target, X_test: Features, y_test: Target) -> float:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            estimator.fit(X_train, y_train)
        scorer = get_scorer(
            self.config.optimization_config.metric.value.sklearn_name)
        return scorer(estimator, X_test, y_test)

    def optimize(self, X: Features, y: Target, override_numeric_features: Optional[List[str]] = None, override_categorical_features: Optional[List[str]] = None) -> AutoResults:
        self._setup_features(X, override_numeric_features,
                             override_categorical_features)
        X_train, X_test, y_train, y_test = self._split_train_test(X, y)

        results = super().optimize(X_train, y_train)

        test_score = self._fit_and_test(
            results.pipeline, X_train, y_train, X_test, y_test)

        return AutoResults(results.pipeline, results.cv_results, test_score)


class AutoVoting(AutoML):
    def __init__(self, config: AutoConfig, test_size: float = 0.3, voting_count: int = 20, max_candidates: int = 60) -> None:
        super().__init__(config, test_size)
        self.voting_count = voting_count
        self.max_candidates = max_candidates

    def _get_voting_estimators(self, pipelines: List[Pipeline]) -> List[Tuple[str, Pipeline]]:
        return [(f'pipeline{i}', pipe) for i, pipe in enumerate(pipelines)]

    def _get_voter(self, pipelines: List[Pipeline]) -> Pipeline:
        voting: VotingClassifier | VotingRegressor = None
        if self.config.problem_type is ProblemType.classification:
            voting = VotingClassifier(
                self._get_voting_estimators(pipelines), voting='soft')
        elif self.config.problem_type is ProblemType.regression:
            voting = VotingRegressor(self._get_voting_estimators(pipelines))
        return make_pipeline(voting)

    def _get_best_candidate(self, current: List[Pipeline], candidates: List[Pipeline], X: Features, y: Target) -> AutoResults:
        best_candidate: Pipeline = None
        best_score: float = None

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            for candidate in candidates:
                candidate_voter = self._get_voter(current + [candidate])
                scores = cross_val_score(
                    candidate_voter, X, y,
                    cv=self.config.optimization_config.cv,
                    n_jobs=self.config.optimization_config.n_jobs,
                    scoring=self.config.optimization_config.metric.value.sklearn_name
                )

                mean_score = np.mean(scores)
                if is_better(self.config.optimization_config.metric, mean_score, best_score):
                    best_candidate = candidate
                    best_score = mean_score

        return AutoResults(best_candidate, best_score)

    def _get_candidates(self) -> List[Pipeline]:
        results = list(self.pipeline_chooser.results)
        sorted_results: List[OptimizationResults] = sort_by_metric(
            results, self.config.optimization_config.metric, key=lambda res: res.evaluation)
        sorted_results = sorted_results[:self.max_candidates]
        return [make_pipeline(results.column_transformer, results.estimator) for results in sorted_results]

    def _compose_estimators(self, results: AutoResults, X: Features, y: Target) -> Tuple[List[Pipeline], float]:
        current = [results.pipeline]
        candidates = self._get_candidates()
        cv_score: float = None
        # the first estimator is already chosen
        for i in range(1, self.voting_count):
            results = self._get_best_candidate(current, candidates, X, y)
            cv_score = results.cv_results
            current += [results.pipeline]
        return current, cv_score

    def optimize(self, X: Features, y: Target, override_numeric_features: Optional[List[str]] = None, override_categorical_features: Optional[List[str]] = None) -> AutoResults:
        best_results = super().optimize(X, y, override_numeric_features,
                                        override_categorical_features)
        X_train, X_test, y_train, y_test = self._split_train_test(X, y)

        estimators, cv_score = self._compose_estimators(
            best_results, X_train, y_train)

        voter = self._get_voter(estimators)
        test_score = self._fit_and_test(
            voter, X_train, y_train, X_test, y_test)

        return AutoResults(voter, cv_score, test_score)


class AutoProgressBar:
    def __init__(self, name: str, auto: AutoPipelineChooser) -> None:
        self.name = name
        self.pipeline = OptimizerProgressBar(
            f'{name} PipelineChooser', auto.pipeline_chooser)
        self.model = OptimizerProgressBar(
            f'{name} ModelChooser', auto.pipeline_chooser.model_chooser)
        self.hyperparameter = OptimizerProgressBar(
            f'{name} HyperparameterOptimizer', auto.pipeline_chooser.model_chooser.optimizer)
