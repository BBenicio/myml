"""
Setup the search space for use of the `auto` module for regression problems.

The `estimators` object contains the search space for regression problems,
with scikit-learn compatible regressors and their respective hyperparameter search spaces.
"""

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from skopt.space import Categorical, Integer, Real
from myml.optimization.search import HyperparameterSearchSpace, ModelSearchSpace


estimators = ModelSearchSpace()
""" Search space for regressors. """

estimators[Ridge()] = HyperparameterSearchSpace(
    alpha=Real(0, 2),
    max_iter=Integer(100, 1000)
)

estimators[RandomForestRegressor()] = HyperparameterSearchSpace(
    n_estimators=Integer(10, 200),
    criterion=Categorical(['squared_error', 'friedman_mse', 'absolute_error']),
    min_samples_split=Integer(2, 200),
    min_samples_leaf=Integer(1, 100)
)

estimators[GradientBoostingRegressor()] = HyperparameterSearchSpace(
    n_estimators=Integer(10, 400),
    min_samples_split=Integer(2, 200),
    min_samples_leaf=Integer(1, 100),
    learning_rate=Real(1e-6, 1e-1, 'log-uniform')
)

estimators[AdaBoostRegressor()] = HyperparameterSearchSpace(
    n_estimators=Integer(10, 200),
)

estimators[KNeighborsRegressor()] = HyperparameterSearchSpace(
    n_neighbors=Integer(3, 50),
    weights=Categorical(['uniform', 'distance'])
)
