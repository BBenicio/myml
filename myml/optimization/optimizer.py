from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline
from skopt.space import Dimension
from skopt.utils import use_named_args
from skopt import gp_minimize
from myml.optimization.metric import Metric, is_better


class HyperparameterOptimizer:
    def __init__(self, estimator: BaseEstimator, metric: Metric, evaluations: int = 100, cv: int = 5, n_jobs: int = 1, seed: Optional[int] = None) -> None:
        self.estimator = estimator
        self.metric = metric
        self.evaluations = evaluations
        self.cv = cv
        self.n_jobs = n_jobs
        self.seed = seed
        self._search_space: List[Dimension] = []
        self.results: List[Dict[str, Any]] = []
        self.preprocess: TransformerMixin = None

    @property
    def search_space(self) -> Dict[str, Dimension]:
        return {dim.name: dim for dim in self._search_space}

    @search_space.setter
    def search_space(self, search_space: Dict[str, Dimension]) -> None:
        self._search_space.clear()
        for name in search_space:
            search_space[name].name = name
            self._search_space.append(search_space[name])
        
    def _translate_metric(self, value: float) -> float:
        if self.metric.value.is_maximize:
            return -value
        elif self.metric.value.is_minimize:
            return value

    def _setup_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
        if self.seed is not None and 'random_state' in vars(self.estimator):
            self.estimator.random_state = self.seed
        self.estimator.set_params(**params)
        est = self.estimator
        if self.preprocess is not None:
            est = make_pipeline(self.preprocess, self.estimator)
        return est

    def get_objective(self, X, y) -> Callable[..., float]:
        @use_named_args(self._search_space)
        def objective(**params) -> float:
            est = self._setup_estimator(params)

            mean_score = np.mean(cross_val_score(est, X, y,
                                 cv=self.cv, n_jobs=self.n_jobs, scoring=self.metric.value.sk_name))
            mean_score = self._translate_metric(mean_score)
            return mean_score

        return objective

    def _update_progress_bar(self, res: Any) -> None:
        self.results.append({'model': self.estimator, 'params': self._make_params(res.x), 'score': res.fun})
        self._best_result = min(res.fun, self._best_result)
        self._progress_bar.set_postfix({'best': self._best_result})
        self._progress_bar.update(1)

    def _make_params(self, params_list: List[Any]) -> Dict[str, Any]:
        params = {}
        for i, x in enumerate(params_list):
            params[self._search_space[i].name] = x
        return params

    def optimize(self, X, y) -> Tuple[Dict[str, Any], float]:
        self.results.clear()
        self._best_result = np.inf
        with tqdm.tqdm(total = self.evaluations, postfix={'best': self._best_result}, desc=f'Optimizing {self.estimator.__class__.__name__}') as self._progress_bar:
            result = gp_minimize(
                self.get_objective(X, y),
                self._search_space,
                n_calls=self.evaluations,
                random_state=self.seed,
                callback=[self._update_progress_bar]
            )

        best_params = self._make_params(result.x)
        best_eval = self._translate_metric(result.fun)

        return best_params, best_eval
    
    def validate(self, best_params: Dict[str, Any], X, y, metrics: List[Metric]) -> Dict[str, float]:
        est = self._setup_estimator(best_params)
        scores = cross_validate(est, X, y, cv=self.cv, n_jobs=self.n_jobs, scoring=[m.value.sk_name for m in metrics])
        summary = {}
        for k in scores:
            if 'test' in k:
                current = scores[k]
                summary[f'mean_validation_{k[5:]}'] = np.mean(current)
                summary[f'std_validation_{k[5:]}'] = np.std(current)

        return summary


class ModelChooser:
    def __init__(self, metric: Metric, evaluations: int = 100, cv: int = 5, n_jobs: int = 1, seed: Optional[int] = None) -> None:
        self.metric = metric
        self.evaluations = evaluations
        self.cv = cv
        self.n_jobs = n_jobs
        self.seed = seed
        self.optimizer = HyperparameterOptimizer(
            BaseEstimator(), metric, evaluations, cv, n_jobs, seed)
        self._search_space: Dict[BaseEstimator, Dict[str, Dimension]] = {}
        self.results: List[Dict[str, Any]] = []
        self.best_params: Dict[BaseEstimator, Dict[str, Any]] = {}

    @property
    def search_space(self) -> Dict[BaseEstimator, Dict[str, Dimension]]:
        return self._search_space

    @search_space.setter
    def search_space(self, search_space: Dict[BaseEstimator, Dict[str, Dimension]]) -> None:
        for estimator in search_space:
            assert isinstance(estimator, BaseEstimator)
            for param in search_space[estimator]:
                assert type(param) is str
                assert isinstance(search_space[estimator][param], Dimension)
        self._search_space = search_space
    
    @property
    def preprocess(self) -> TransformerMixin:
        return self.optimizer.preprocess
    
    @preprocess.setter
    def preprocess(self, value: TransformerMixin):
        self.optimizer.preprocess = value

    def optimize(self, X, y) -> Tuple[BaseEstimator, Dict[str, Any], float]:
        self.results.clear()
        best_estimator = None
        best_params = None
        best_result = None
        with tqdm.tqdm(self._search_space, desc='ModelChooser', postfix={'best': best_result}) as pbar:
            for estimator in pbar:
                self.optimizer.estimator = estimator
                self.optimizer.search_space = self._search_space[estimator]
                params, result = self.optimizer.optimize(X, y)
                self.best_params[estimator] = params
                self.results.extend(self.optimizer.results)
                if is_better(self.metric, result, best_result):
                    best_estimator = estimator
                    best_params = params
                    best_result = result
                pbar.set_postfix({'best': best_result})

        return best_estimator, best_params, best_result


class PipelineChooser(ModelChooser):
    def __init__(self, metric: Metric, evaluations: int = 100, cv: int = 5, n_jobs: int = 1, seed: Optional[int] = None) -> None:
        super().__init__(metric, evaluations, cv, n_jobs, seed)
        self._pipeline_search_space: List[List[Tuple[TransformerMixin, List[str]]]] = {}
        self.pipeline_results: List[Dict[str, Any]] = []
    
    @property
    def pipeline_search_space(self) -> List[List[Tuple[TransformerMixin, List[str]]]]:
        return self._pipeline_search_space

    @pipeline_search_space.setter
    def pipeline_search_space(self, pipeline_search_space: List[List[Tuple[TransformerMixin, List[str]]]]) -> None:
        for option in pipeline_search_space:
            for transformer, _ in option:
                assert isinstance(transformer, TransformerMixin)
        self._pipeline_search_space = pipeline_search_space
    
    def optimize(self, X, y) -> Tuple[ColumnTransformer, BaseEstimator, Dict[str, Any], float]:
        self.pipeline_results.clear()
        best_ct = None
        best_estimator = None
        best_params = None
        best_result = None
        with tqdm.tqdm(self._pipeline_search_space, desc='PipelineChooser', postfix={'best': best_result}) as pbar:
            for option in pbar:
                ct = make_column_transformer(*option)
                self.preprocess = ct
                est, params, result = super().optimize(X, y)
                est.set_params(**params)
                pipe = make_pipeline(ct, est)
                for res in self.results:
                    self.pipeline_results.append({'model': pipe, 'params': res['params'], 'score': res['score']})
                if is_better(self.metric, result, best_result):
                    best_ct = ct
                    best_estimator = est
                    best_params = params
                    best_result = result
                pbar.set_postfix({'best': best_result})
        
        return best_ct, best_estimator, best_params, best_result
    