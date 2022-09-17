from typing import Any, Dict, List, Optional, Tuple
from myml.optimization.metric import Metric
from myml.optimization.optimizer import PipelineChooser
from myml.utils import DataType, ProblemType
from myml.models import classifiers, regressors
from myml.preprocessors import categorical_preprocessors, numeric_preprocessors
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import get_scorer
import numpy as np
import pandas as pd


class AutoPipelineChooser:
    def __init__(self, problem_type: ProblemType, metric: Metric, evaluations: int = 100, cv: int = 5, n_jobs: int = 1, seed: Optional[int] = None) -> None:
        self.problem_type = problem_type
        self.metric = metric
        self.cv = cv
        self.n_jobs = n_jobs
        self.seed = seed

        self.pipeline_chooser = PipelineChooser(metric, evaluations, cv, n_jobs, seed)
        self.data_types: Dict[DataType, List[Any]] = {}
    
    @property
    def results(self) -> List[Dict[str, Any]]:
        return self.pipeline_chooser.pipeline_results
    
    @property
    def categorical_features(self) -> List[Any]:
        return self.data_types[DataType.categorical] if DataType.categorical in self.data_types else []
    
    @categorical_features.setter
    def categorical_features(self, features: List[Any]):
        self.data_types[DataType.categorical] = features
    
    # @property
    # def ordinal_features(self) -> List[Any]:
    #     return self.data_types[DataType.ordinal] if DataType.ordinal in self.data_types else []
    
    # @ordinal_features.setter
    # def ordinal_features(self, features: List[Any]):
    #     self.data_types[DataType.ordinal] = features
    
    @property
    def numeric_features(self) -> List[Any]:
        return self.data_types[DataType.numeric] if DataType.numeric in self.data_types else []
    
    @numeric_features.setter
    def numeric_features(self, features: List[Any]):
        self.data_types[DataType.numeric] = features

    # @property
    # def textual_features(self) -> List[Any]:
    #     return self.data_types[DataType.textual] if DataType.textual in self.data_types else []
    
    # @textual_features.setter
    # def textual_features(self, features: List[Any]):
    #     self.data_types[DataType.textual] = features

    def _assemble_search_space(self, X, y):
        if self.problem_type == ProblemType.classification:
            self.pipeline_chooser.search_space = classifiers
        elif self.problem_type == ProblemType.regression:
            self.pipeline_chooser.search_space = regressors
        
        def increment_pipeline(features: List, preprocessors: List, pipelines: List) -> List:
            new_pipelines = []
            if len(features) > 0:
                options = [(preprocessor, features) for preprocessor in preprocessors]
                for pipe in pipelines:
                    for op in options:
                        new_pipelines.append(pipe + [op])
            else:
                new_pipelines = pipelines

            return new_pipelines
        
        features_have_nan = np.any(np.isnan(X))
        pipelines = [[SimpleImputer()]] if features_have_nan else [[]]
        pipelines = increment_pipeline(self.numeric_features, numeric_preprocessors, pipelines)
        pipelines = increment_pipeline(self.categorical_features, categorical_preprocessors, pipelines)
        # pipelines = increment_pipeline(self.ordinal_features, ordinal_preprocessors, pipelines)

        self.pipeline_chooser.pipeline_search_space = pipelines
        
    def optimize(self, X, y) -> Tuple[Pipeline, float]:
        self._assemble_search_space(X, y)
        ct, model, params, result = self.pipeline_chooser.optimize(X, y)
        model.set_params(**params)
        return make_pipeline(ct, model), result


class AutoML(AutoPipelineChooser):
    def __init__(self, problem_type: ProblemType, metric: Metric, test_size: float = 0.3, evaluations: int = 100, cv: int = 5, n_jobs: int = 1, seed: Optional[int] = None) -> None:
        super().__init__(problem_type, metric, evaluations, cv, n_jobs, seed)
        self.test_size = test_size
    
    def optimize(self, X: pd.DataFrame, y: pd.DataFrame, voting: int = 0, override_numeric_features: Optional[List[str]] = None, override_categorical_features: Optional[List[str]] = None) -> Tuple[Pipeline, float, float]:
        if override_numeric_features is None:
            self.numeric_features = X.select_dtypes(include=['number']).columns.to_list()
        else:
            self.numeric_features = override_numeric_features
        if override_categorical_features is None:
            self.categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.to_list()
        else:
            self.categorical_features = override_categorical_features
        
        if self.problem_type == ProblemType.classification:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.seed, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.seed)

        pipeline, result = super().optimize(X_train, y_train)
        if voting > 1:
            voter_pipeline, voter_result = self._make_voter(X, y, top=voting)
        
            if voter_result > result:
                pipeline = voter_pipeline
                result = voter_result

        pipeline.fit(X_train, y_train)
        scorer = get_scorer(self.metric.value.sk_name)
        test_score = scorer(pipeline, X_test, y_test)
        
        return pipeline, result, test_score

    def _make_voter(self, X: pd.DataFrame, y: pd.DataFrame, top: int = 5) -> Tuple[VotingClassifier | VotingRegressor, float]:
        sorted_results = sorted(self.results, key=lambda res: res['score'])[:top]
        top_estimators = []
        for i, res in enumerate(sorted_results):
            est = res['model']
            name = est.steps[-1][0]
            top_estimators.append((f'{name}_{i}', est))
        top_estimators = [(f'{res["model"].steps[-1][0]}_{i}', res['model']) for i, res in enumerate(sorted_results)]
        if self.problem_type == ProblemType.classification:
            voter = VotingClassifier(top_estimators, voting='soft')
        elif self.problem_type == ProblemType.regression:
            voter = VotingRegressor(top_estimators)
        
        return voter, np.mean(cross_val_score(voter, X, y, scoring=self.metric.value.sk_name, cv=self.cv, n_jobs=self.n_jobs))
