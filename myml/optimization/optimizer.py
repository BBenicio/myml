"""
Optimize hyperparameters, algorithms and data preprocessing pipelines.

Using the `PipelineChooser` it is possible to find the best combination
of data pipeline and algorithm with optimized hyperparameters for the
training dataset.

With `CashOptimizer`, optimization of algorithms and hyperparameters is
run in a single optimization process, wasting fewer resources and still
exploring the search space in a smart way.

`HyperparameterOptimizer` is for when the problem requires only
hyperparameter optimization. It takes a search space for a given estimator
and finds the best hyperparameter configurations using Bayseian
Optimization.
"""

from abc import ABC, abstractmethod
import tqdm
import numpy as np
from copy import copy
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from skopt.utils import use_named_args
from skopt.space import Categorical
from skopt import gp_minimize, forest_minimize
from myml.optimization.metric import Metric, is_better, translate_metric
from myml.optimization.search import HyperparameterSearchSpace, ModelSearchSpace, PipelineSearchSpace, SearchSpace
from myml.utils import Features, Target


class OptimizationResults(NamedTuple):
    """
    Results of an optimization run.

    Parameters
    ----------
    evaluation : float
        Score achieved on the metric being optimized.
    hyperparameters : dict of {str: Any}, default {}
        Hyperparameters settings for the estimator
    estimator : sklearn.base.BaseEstimator, optional
        Estimator used on the optimization
    column_transformer : sklearn.compose.ColumnTransformer, optional
        Preprocessing step used on the optimization

    Attributes
    ----------
    evaluation : float
        Score achieved on the metric being optimized.
    hyperparameters : dict of {str: Any}
        Hyperparameters settings for the estimator
    estimator : sklearn.base.BaseEstimator
        Estimator used on the optimization
    column_transformer : sklearn.compose.ColumnTransformer
        Preprocessing step used on the optimization
    """

    evaluation: float
    hyperparameters: Dict[str, Any] = {}
    estimator: BaseEstimator = None
    column_transformer: ColumnTransformer = None


class OptimizationConfig(NamedTuple):
    """
    Configuration for the optimizer.

    Parameters
    ----------
    metric : Metric
        Metric to optimize
    evaluations : int, default 100
        Number of evaluations to perform on the optimization, more
        evaluations usually mean a better result, but more costly.
    cv : int, default 5
        Number of folds to evaluate using k-fold cross-validation.
    n_jobs : int, default 1
        Number of computing processors to use.
    seed: int, optional
        Seed for the random number generator. Set when you need to
        reproduce the results.

    Attributes
    ----------
    metric : Metric
        Metric to optimize
    evaluations : int
        Number of evaluations to perform on the optimization, more
        evaluations usually mean a better result, but more costly.
    cv : int
        Number of folds to evaluate using k-fold cross-validation.
    n_jobs : int
        Number of computing processors to use.
    seed: int
        Seed for the random number generator. Set when you need to
        reproduce the results.
    """
    metric: Metric
    evaluations: int = 100
    cv: int = 5
    n_jobs: int = 1
    seed: Optional[int] = None


class Optimizer(ABC):
    """
    An optimizer of machine learning models.

    Attributes
    ----------
    results : Iterator[OptimizationResults]
        Results from each step of optimization
    search_space : SearchSpace
        Search space of the optimizer.
    """

    progress_callback: Callable[[Dict[str, Any], int], None] = None
    """ Function to call on each optimization step to report partial results. """

    @property
    @abstractmethod
    def results(self) -> Iterator[OptimizationResults]:
        """ Results from each step of optimization """

    @property
    @abstractmethod
    def search_space(self) -> SearchSpace:
        """ Search space of the optimizer. """

    @search_space.setter
    @abstractmethod
    def search_space(self, search_space: SearchSpace) -> None:
        """ Search space of the optimizer. """

    @abstractmethod
    def optimize(self, X: Features, y: Target) -> OptimizationResults:
        """ Optimize given the features and targets. """

    def _update_progress(self, data: Dict[str, Any] = None, total: int = 0):
        """
        Report partial progress to the callback registered.

        Parameters
        ----------
        data : dict of {str: Any}, optional
            The data to send on the partial report.
        total : int, default 0
            The total count of optimization steps that will be run.
        """
        if self.progress_callback is not None:
            self.progress_callback(data, total)


class HyperparameterOptimizer(Optimizer):
    """
    An optimizer for algorithm hyperparameters.

    Given an algorithm and a defined search space of hyperparameters to
    evaluate, use Bayesian optimization to find a good set of
    hyperparameters that optimize some metric.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        The algorithm to optimize the hyperparameters.
    config : OptimizationConfig
        Configuration for the optimization to run.

    Attributes
    ----------
    estimator : sklearn.base.BaseEstimator
        The algorithm to optimize the hyperparameters.
    config : OptimizationConfig
        Configuration for the optimization to run.
    results : Iterator[OptimizationResults]
        Results from each step of optimization
    search_space : HyperparameterSearchSpace
        Search space of the optimizer.
    preprocess : sklearn.base.TransformerMixin
        Preprocessor for the features before passing to the model.
    """

    def __init__(self, estimator: BaseEstimator, config: OptimizationConfig) -> None:
        self.estimator = estimator
        self.config = config
        self._search_space: HyperparameterSearchSpace = HyperparameterSearchSpace()
        self.preprocess: TransformerMixin = None
        self._results: List[OptimizationResults] = []

    @property
    def results(self) -> Iterator[OptimizationResults]:
        yield from self._results

    @property
    def search_space(self) -> HyperparameterSearchSpace:
        return self._search_space

    @search_space.setter
    def search_space(self, search_space: HyperparameterSearchSpace) -> None:
        self._search_space = search_space

    def _setup_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
        """
        Get the estimator set with the hyperparameters and preprocessor.

        Parameters
        ----------
        params : dict of {str: Any}
            Hyperparameters got from an optimization step.

        Returns
        -------
        sklearn.base.BaseEstimator
            The estimator set with the given hyperparameters and
            preprocessor.
        """
        self.estimator.random_state = self.config.seed
        self.estimator.set_params(**params)
        return self.estimator if self.preprocess is None else make_pipeline(self.preprocess, self.estimator)

    def _get_objective(self, X: Features, y: Target) -> Callable[..., float]:
        """
        Get the objective function for the optimization.

        Parameters
        ----------
        X : Features
            Features to use for model training
        y : Target
            Target values for model training

        Returns
        -------
        Callable
            Objective function compatible with sckit-optimize.
        """
        @use_named_args(list(self._search_space))
        def objective(**params) -> float:
            estimator = self._setup_estimator(params)

            scores = cross_val_score(
                estimator, X, y,
                cv=self.config.cv,
                n_jobs=self.config.n_jobs,
                scoring=self.config.metric.value.sklearn_name
            )

            mean_score = np.mean(scores)
            mean_score = translate_metric(self.config.metric, mean_score)
            return mean_score

        return objective

    def _optimization_step(self, result: Any) -> None:
        """
        Store and report the results from each optimization step.

        Parameters
        ----------
        result : scipy.optimize.OptimizeResult
            Result from the optimization step.
        """
        self._results.append(self._make_results(result))
        self._best_result = min(result.fun, self._best_result)
        self._update_progress({'best': self._best_result},
                              self.config.evaluations)

    def _make_params(self, params: List[Any]) -> Dict[str, Any]:
        """
        Transform the hyperparameters from a list to a dictionary.

        Parameters
        ----------
        params : list
            List of hyperparameters settings to transform.

        Returns
        -------
        dict of {str: Any}
            Hyperparameters in a dictionary of `{name: value}`
        """
        transformed = {}
        for dimension, value in zip(self._search_space, params):
            transformed[dimension.name] = value
        return transformed

    def _make_results(self, result: Any) -> OptimizationResults:
        """
        Transform the results into `OptimizationResults`.

        Parameters
        ----------
        result : scipy.optimize.OptimizeResult
            Result from the scikit-optimize optimization run.

        Returns
        -------
        OptimizationResults
            Results from the hyperparameter optimization.
        """
        params = self._make_params(result.x)
        self.estimator.set_params(**params)

        return OptimizationResults(
            evaluation=translate_metric(self.config.metric, result.fun),
            hyperparameters=params,
            estimator=self.estimator,
            column_transformer=self.preprocess if isinstance(
                self.preprocess, ColumnTransformer) else None
        )

    def optimize(self, X: Features, y: Target) -> OptimizationResults:
        """
        Optimize the hyperparameters on the given dataset.

        Parameters
        ----------
        X : Features
            Features to use for model training
        y : Target
            Target values for model training

        Returns
        -------
        OptimizationResults
            Results from the hyperparameter optimization.
        """
        self._best_result = np.inf
        result = gp_minimize(
            self._get_objective(X, y),
            self._search_space,
            n_calls=self.config.evaluations,
            random_state=self.config.seed,
            callback=[self._optimization_step]
        )

        return self._make_results(result)


class ModelChooser(Optimizer):
    """
    Optimizer for choosing algorithms and optimizing hyperparameters.

    For a set of machine learning algorithms, optimize the hyperparameters
    on each of them to compare the algorithms and find the best performing
    model on the data according to some metric.

    Parameters
    ----------
    config : OptimizationConfig
        Configuration for the optimization to run.
    optimizer : HyperparameterOptimizer, optional
        Hyperparameter optimizer to use.

    Attributes
    ----------
    config : OptimizationConfig
        Configuration for the optimization to run.
    optimizer : HyperparameterOptimizer
        Hyperparameter optimizer to use.
    results : Iterator[OptimizationResults]
        Results from each step of optimization
    search_space : ModelSearchSpace
        Search space of the optimizer.
    preprocess : sklearn.base.TransformerMixin
        Preprocessor for the features before passing to the model.
    """

    def __init__(self, config: OptimizationConfig, optimizer: Optional[HyperparameterOptimizer] = None) -> None:
        self.config = config
        self.optimizer = optimizer if optimizer is not None else HyperparameterOptimizer(
            BaseEstimator(), config)
        self._search_space: ModelSearchSpace = ModelSearchSpace()

    @property
    def results(self) -> Iterator[OptimizationResults]:
        yield from self.optimizer.results

    @property
    def search_space(self) -> ModelSearchSpace:
        return self._search_space

    @search_space.setter
    def search_space(self, search_space: ModelSearchSpace) -> None:
        self._search_space = search_space

    @property
    def preprocess(self) -> TransformerMixin:
        return self.optimizer.preprocess

    @preprocess.setter
    def preprocess(self, value: TransformerMixin):
        """ Preprocessor for the features before passing to the model. """
        self.optimizer.preprocess = value

    def _optimize_hyperparameters(self, estimator: BaseEstimator, X: Features, y: Target) -> OptimizationResults:
        """
        Optimize the hyperparameters of the given estimator.

        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator
            estimator to optimze the hyperparameters
        X : Features
            Features to use for model training
        y : Target
            Target values for model training
        Returns
        -------
        OptimizationResults
            Results from the hyperparameter optimization.
        """
        self.optimizer.estimator = estimator
        self.optimizer.search_space = self._search_space[estimator]
        return self.optimizer.optimize(X, y)

    def optimize(self, X: Features, y: Target) -> OptimizationResults:
        """
        Choose the best algorithm and hyperparameters for the data.

        Parameters
        ----------
        X : Features
            Features to use for model training
        y : Target
            Target values for model training
        Returns
        -------
        OptimizationResults
            Results from the algorithm optimization.
        """
        best_results: OptimizationResults = OptimizationResults(
            evaluation=None)

        for estimator in self._search_space:
            optimization_results = self._optimize_hyperparameters(
                estimator, X, y)

            if is_better(self.config.metric, optimization_results.evaluation, best_results.evaluation):
                best_results = optimization_results

            self._update_progress(
                {'best': best_results.evaluation}, len(self._search_space))

        return best_results


class CashOptimizer(ModelChooser):
    """
    Combined Algorithm Selection and Hyperparameter optimization (CASH).

    Use bayesian optimization on algorithms and hyperparameters in a
    combined manner, searching for the best algorithm-hyperparameter
    combination in the given search space.

    Parameters
    ----------
    config : OptimizationConfig
        Configuration for the optimization to run.

    Attributes
    ----------
    config : OptimizationConfig
        Configuration for the optimization to run.
    results : Iterator[OptimizationResults]
        Results from each step of optimization
    search_space : ModelSearchSpace
        Search space of the optimizer.
    preprocess : sklearn.base.TransformerMixin
        Preprocessor for the features before passing to the model.
    """

    def __init__(self, config: OptimizationConfig) -> None:
        super().__init__(config, None)
        self._preprocess: TransformerMixin = None
        self._results: List[OptimizationResults] = []
        self._hyperparameter_search_space: HyperparameterSearchSpace = None

    @property
    def results(self) -> Iterator[OptimizationResults]:
        yield from self._results

    @property
    def preprocess(self) -> TransformerMixin:
        return self._preprocess

    @preprocess.setter
    def preprocess(self, value: TransformerMixin):
        self._preprocess = value

    def _get_algorithm_name(self, estimator: BaseEstimator) -> str:
        """
        Get the name that represents an algorithm.

        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator
            The estimator from which to extract the name.

        Returns
        -------
        str
            The name for the algorithm.
        """
        return estimator.__class__.__name__.lower()

    def _get_search_space(self) -> HyperparameterSearchSpace:
        """
        Get the search space as hyperparameters only.

        Consider the algorithm as a hyperparameter and estabilish a
        relationship between hyperparameters and their respective
        algorithms.

        Returns
        -------
        HyperparameterSeachSpace
            The search space for the optimization as hyperparameters.
        """
        total_space = HyperparameterSearchSpace()
        total_space['[estimator]'] = Categorical(
            list(self.search_space.keys()))
        for estimator in self.search_space:
            search_space = self.search_space[estimator]
            for dimension in search_space:
                dim = copy(dimension)
                name = f'{self._get_algorithm_name(estimator)}/{dim.name}'
                total_space[name] = dim

        return total_space

    def _filter_relevant_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter the hyperparameters to keep only relevant ones.

        Check the current estimator selected and remove any
        hyperparameters that are not related.

        Parameters
        ----------
        params : dict of {str: Any}
            Hyperparameters and their values found in and optimization run.

        Returns
        -------
        dict of {str: Any}
            The relevant hyperparameters and values for the current
            estimator.
        """
        estimator = params['[estimator]']
        relevant_params = {k.split(
            '/')[1]: params[k] for k in params if self._get_algorithm_name(estimator) in k}
        return relevant_params

    def _get_estimator_from_params(self, params: Dict[str, Any]) -> BaseEstimator:
        """
        Retrieve the estimator from the hyperparameters.

        Get the estimator and set the hyperparameters needed to configure
        the estimator properly for training.

        Parameters
        ----------
        params : dict of {str: Any}
            Hyperparameters and their values found in and optimization run.

        Returns
        -------
        sklearn.base.BaseEstimator
            Estimator configured with the given hyperparameters.
        """
        estimator = params['[estimator]']
        relevant_params = self._filter_relevant_params(params)
        estimator.set_params(**relevant_params)
        return estimator

    def _setup_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
        """
        Setup the estimator and data pipeline from the hyperparameters.

        Parameters
        ----------
        params : dict of {str: Any}
            Hyperparameters and their values found in and optimization run.

        Returns
        -------
        sklearn.base.BaseEstimator
            Estimator configured with the given hyperparameters.        
        """
        estimator = self._get_estimator_from_params(params)
        estimator.random_state = self.config.seed
        return estimator if self.preprocess is None else make_pipeline(self.preprocess, estimator)

    def _get_objective(self, X: Features, y: Target) -> Callable[..., float]:
        """
        Get the objective function for the optimization.

        Parameters
        ----------
        X : Features
            Features to use for model training
        y : Target
            Target values for model training

        Returns
        -------
        Callable
            Objective function compatible with sckit-optimize.
        """
        @use_named_args(list(self._hyperparameter_search_space))
        def objective(**params) -> float:
            estimator = self._setup_estimator(params)

            scores = cross_val_score(
                estimator, X, y,
                cv=self.config.cv,
                n_jobs=self.config.n_jobs,
                scoring=self.config.metric.value.sklearn_name
            )

            mean_score = np.mean(scores)
            mean_score = translate_metric(self.config.metric, mean_score)
            return mean_score

        return objective

    def _optimization_step(self, result: Any) -> None:
        """
        Store and report the results from each optimization step.

        Parameters
        ----------
        result : scipy.optimize.OptimizeResult
            Result from the optimization step.
        """
        self._results.append(self._make_results(result))
        self._best_result = min(result.fun, self._best_result)
        self._update_progress({'best': self._best_result},
                              self.config.evaluations)

    def _make_params(self, params_list: List[Any]) -> Dict[str, Any]:
        """
        Transform the hyperparameters from a list to a dictionary.

        Parameters
        ----------
        params : list
            List of hyperparameters settings to transform.

        Returns
        -------
        dict of {str: Any}
            Hyperparameters in a dictionary of `{name: value}`
        """
        params = {}
        for dimension, value in zip(self._hyperparameter_search_space, params_list):
            params[dimension.name] = value
        return params

    def _make_results(self, result: Any) -> OptimizationResults:
        """
        Transform the results into `OptimizationResults`.

        Parameters
        ----------
        result : scipy.optimize.OptimizeResult
            Result from the scikit-optimize optimization run.

        Returns
        -------
        OptimizationResults
            Results from the hyperparameter optimization.
        """
        params = self._make_params(result.x)
        estimator = self._get_estimator_from_params(params)
        relevant_params = self._filter_relevant_params(params)

        return OptimizationResults(
            evaluation=translate_metric(self.config.metric, result.fun),
            hyperparameters=relevant_params,
            estimator=estimator,
            column_transformer=self.preprocess if isinstance(
                self.preprocess, ColumnTransformer) else None
        )

    def optimize(self, X: Features, y: Target) -> OptimizationResults:
        """
        Optimize the algorithm and hyperparameters on the given dataset.

        Parameters
        ----------
        X : Features
            Features to use for model training
        y : Target
            Target values for model training

        Returns
        -------
        OptimizationResults
            Results from the hyperparameter and algorithm optimization.
        """
        self._hyperparameter_search_space = self._get_search_space()
        self._best_result = np.inf
        result = forest_minimize(
            self._get_objective(X, y),
            self._hyperparameter_search_space,
            n_calls=self.config.evaluations,
            random_state=self.config.seed,
            callback=[self._optimization_step]
        )

        return self._make_results(result)


class PipelineChooser(Optimizer):
    """
    Optimizer for choosing data pipelines and algorithms.

    For a set of data preprocessors, find the algorithms that work best
    with each combination of data preprocessing pipeline to get the
    best combination of data pipeline, algorithm and hyperparameters.

    In order to optimize the pipeline, it is also needed to estabilish the
    search space for the algorithm and hyperparameter optimization by the
    `PipelineChooser.model_chooser` attribute.

    Parameters
    ----------
    config : OptimizationConfig
        Configuration for the optimization to run.
    model_chooser : ModelChooser, optional
        Model chooser to use for algorithm selection and hyperparameter
        optimization.

    Attributes
    ----------
    config : OptimizationConfig
        Configuration for the optimization to run.
    model_chooser : ModelChooser, optional
        Model chooser to use for algorithm selection and hyperparameter
        optimization.
    results : Iterator[OptimizationResults]
        Results from each step of optimization
    search_space : PipelineSearchSpace
        Search space of the optimizer.
    """

    def __init__(self, config: OptimizationConfig, model_chooser: Optional[ModelChooser] = None) -> None:
        self.config = config
        self.model_chooser = model_chooser if model_chooser is not None else ModelChooser(
            config)
        self._search_space: PipelineSearchSpace = PipelineSearchSpace()

    @property
    def results(self) -> Iterator[OptimizationResults]:
        yield from self.model_chooser.results

    @property
    def search_space(self) -> PipelineSearchSpace:
        return self._search_space

    @search_space.setter
    def search_space(self, search_space: PipelineSearchSpace) -> None:
        self._search_space = search_space

    def _optimize_models(self, ct: ColumnTransformer, X: Features, y: Target) -> OptimizationResults:
        """
        Optimize the algorithm and hyperparameters for a data pipeline.

        Parameters
        ----------
        ct : sklearn.compose.ColumnTransformer
            Preprocessing step used on the optimization
        X : Features
            Features to use for model training
        y : Target
            Target values for model training

        Returns
        -------
        OptimizationResults
            Results from the hyperparameter and algorithm optimization.
        """
        self.model_chooser.preprocess = ct
        results = self.model_chooser.optimize(X, y)
        return OptimizationResults(
            evaluation=results.evaluation,
            hyperparameters=results.hyperparameters,
            estimator=results.estimator,
            column_transformer=ct
        )

    def optimize(self, X: Features, y: Target) -> OptimizationResults:
        """
        Optimize the data pipeline and algorithm on the given dataset.

        Parameters
        ----------
        X : Features
            Features to use for model training
        y : Target
            Target values for model training

        Returns
        -------
        OptimizationResults
            Results from the data pipeline and algorithm optimization.
        """
        best_results: OptimizationResults = OptimizationResults(
            evaluation=None)

        for option in self._search_space:
            ct = make_column_transformer(*option)
            optimization_results = self._optimize_models(ct, X, y)

            if is_better(self.config.metric, optimization_results.evaluation, best_results.evaluation):
                best_results = optimization_results

            self._update_progress(
                {'best': best_results.evaluation}, len(self._search_space))

        return best_results


class OptimizerProgressBar:
    """
    A progress bar for an optimizer.

    Attach a progress bar to any optimizer to keep track of the time
    running and estimate time remaining to finish the optimization along
    with updated metrics.

    Parameters
    ----------
    name : str
        The name to display on the progress bar.
    optimizer : Optimizer
        The optimizer to attatch the progress bar.

    Attributes
    ----------
    name : str
        The name to display on the progress bar.
    pbar : tqdm.tqdm
        Progress bar handle.
    """

    def __init__(self, name: str, optimizer: Optimizer) -> None:
        self.name = name
        optimizer.progress_callback = self._callback
        self.pbar: tqdm.tqdm = None
        self._current_iteration = 0

    def _initialize_progress_bar(self, total: int) -> None:
        """
        Initialize the progress bar if necessary.

        Parameters
        ----------
        total : int
            Number of iterations that will occur.
        """
        if self.pbar is None:
            self.pbar = tqdm.tqdm(total=total, desc=self.name)
            self._current_iteration = 0

    def _show_data(self, data: Dict[str, Any]) -> None:
        """
        Put data on the progress bar display.

        Parameters
        ----------
        data : dict of {str: Any}
            Data to show on the progress bar, each item will appear as
            `key=value` on the display.
        """
        if data is not None:
            self.pbar.set_postfix(data)

    def _update_progress(self) -> None:
        """ Register an iteration to update the progress bar. """
        self._current_iteration += 1
        self.pbar.update(1)

    def _ended(self, total: int) -> bool:
        """
        Check if the final iteration was reached.

        Parameters
        ----------
        total : int
            Number of iterations to occur.

        Returns
        -------
        bool
            `True` when the total number of iterations was reached.
        """
        return self._current_iteration >= total

    def _close_progress_bar_on_end(self, total: int) -> None:
        """
        Finish the progress bar if it has ended.

        Parameters
        ----------
        total : int
            Number of iterations to occur.
        """
        if self._ended(total):
            self._current_iteration = 0
            self.pbar.close()
            self.pbar = None

    def _callback(self, data: Dict[str, Any], total: int) -> None:
        """
        Receive a partial report from the optimizer.

        Parameters
        ----------
        data : dict of {str: Any}
            The data to received from the partial report.
        total : int
            The total count of optimization steps.
        """
        self._initialize_progress_bar(total)
        self._show_data(data)
        self._update_progress()
        self._close_progress_bar_on_end(total)
