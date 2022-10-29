from abc import ABC, abstractmethod
import tqdm
import numpy as np
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from skopt.utils import use_named_args
from skopt import gp_minimize
from myml.optimization.metric import Metric, is_better, translate_metric
from myml.optimization.search import HyperparameterSearchSpace, ModelSearchSpace, PipelineSearchSpace, SearchSpace
from myml.utils import Features, Target


class OptimizationResults(NamedTuple):
    evaluation: float
    hyperparameters: Dict[str, Any] = {}
    estimator: BaseEstimator = None
    column_transformer: ColumnTransformer = None


class OptimizationConfig(NamedTuple):
    metric: Metric
    evaluations: int = 100
    cv: int = 5
    n_jobs: int = 1
    seed: Optional[int] = None


class Optimizer(ABC):
    progress_callback: Callable[[Dict[str, Any], int], None] = None

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
        if self.progress_callback is not None:
            self.progress_callback(data, total)


class HyperparameterOptimizer(Optimizer):
    def __init__(self, estimator: BaseEstimator, config: OptimizationConfig) -> None:
        self.estimator = estimator
        self.config = config
        self._search_space: HyperparameterSearchSpace = HyperparameterSearchSpace()
        self.preprocess: TransformerMixin = None

    @property
    def search_space(self) -> HyperparameterSearchSpace:
        return self._search_space

    @search_space.setter
    def search_space(self, search_space: HyperparameterSearchSpace) -> None:
        self._search_space = search_space

    def _setup_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
        self.estimator.random_state = self.config.seed
        self.estimator.set_params(**params)
        return self.estimator if self.preprocess is None else make_pipeline(self.preprocess, self.estimator)

    def _get_objective(self, X, y) -> Callable[..., float]:
        @use_named_args(list(self._search_space))
        def objective(**params) -> float:
            est = self._setup_estimator(params)

            mean_score = np.mean(cross_val_score(est, X, y,
                                 cv=self.config.cv, n_jobs=self.config.n_jobs, scoring=self.config.metric.value.sk_name))
            mean_score = translate_metric(self.config.metric, mean_score)
            return mean_score

        return objective

    def _optimization_step(self, res: Any) -> None:
        self._best_result = min(res.fun, self._best_result)
        self._update_progress({'best': self._best_result}, self.config.evaluations)

    def _make_params(self, params_list: List[Any]) -> Dict[str, Any]:
        params = {}
        for dim, val in zip(self._search_space, params_list):
            params[dim.name] = val
        return params

    def optimize(self, X: Features, y: Target) -> OptimizationResults:
        self._best_result = np.inf
        result = gp_minimize(
            self._get_objective(X, y),
            self._search_space,
            n_calls=self.config.evaluations,
            random_state=self.config.seed,
            callback=[self._optimization_step]
        )

        return OptimizationResults(evaluation=translate_metric(self.config.metric, result.fun), hyperparameters=self._make_params(result.x), estimator=self.estimator)


class ModelChooser(Optimizer):
    def __init__(self, config: OptimizationConfig, optimizer: Optional[HyperparameterOptimizer] = None) -> None:
        self.config = config
        self.optimizer = optimizer if optimizer is not None else HyperparameterOptimizer(BaseEstimator(), config)
        self._search_space: ModelSearchSpace = ModelSearchSpace()

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
        self.optimizer.preprocess = value

    def _optimize_hyperparameters(self, estimator: BaseEstimator, X: Features, y: Target) -> OptimizationResults:
        self.optimizer.estimator = estimator
        self.optimizer.search_space = self._search_space[estimator]
        return self.optimizer.optimize(X, y)

    def optimize(self, X: Features, y: Target) -> OptimizationResults:
        best_results: OptimizationResults = OptimizationResults(evaluation=None)
        
        for estimator in self._search_space:
            optimization_results = self._optimize_hyperparameters(estimator, X, y)
            
            if is_better(self.config.metric, optimization_results.evaluation, best_results.evaluation):
                best_results = optimization_results

            self._update_progress({'best': best_results.evaluation}, len(self._search_space))

        return best_results


class PipelineChooser(Optimizer):
    def __init__(self, config: OptimizationConfig, model_chooser: Optional[ModelChooser] = None) -> None:
        self.config = config
        self.model_chooser = model_chooser if model_chooser is not None else ModelChooser(config)
        self._search_space: PipelineSearchSpace = PipelineSearchSpace()
    
    @property
    def search_space(self) -> PipelineSearchSpace:
        return self._search_space

    @search_space.setter
    def search_space(self, search_space: PipelineSearchSpace) -> None:
        self._search_space = search_space
    
    def _optimize_models(self, ct: ColumnTransformer, X: Features, y: Target) -> OptimizationResults:
        self.model_chooser.preprocess = ct
        results = self.model_chooser.optimize(X, y)
        return OptimizationResults(evaluation=results.evaluation, hyperparameters=results.hyperparameters, estimator=results.estimator, column_transformer=ct)
    
    def optimize(self, X: Features, y: Target) -> OptimizationResults:
        best_results: OptimizationResults = OptimizationResults(evaluation=None)

        for option in self._search_space:
            ct = make_column_transformer(*option)
            optimization_results = self._optimize_models(ct, X, y)

            if is_better(self.config.metric, optimization_results.evaluation, best_results.evaluation):
                best_results = optimization_results

            self._update_progress({'best': best_results.evaluation}, len(self._search_space))
        
        return best_results
    

class OptimizerProgressBar:
    def __init__(self, name: str, optimizer: Optimizer) -> None:
        self.name = name
        optimizer.progress_callback = self._callback
        self.pbar: tqdm.tqdm = None
        self.i = 0
    
    def _callback(self, data: Dict[str, Any], total: int) -> None:
        self.i += 1
        if self.pbar is None:
            self.pbar = tqdm.tqdm(total=total, desc=self.name)
        
        if data is not None:
            self.pbar.set_postfix(data)

        self.pbar.update(1)
        
        if self.i == total:
            self.i = 0
            self.pbar.close()
            self.pbar = None
