"""
Setup the search space for use of the `auto` module for regression problems.

The `estimators` object contains the search space for regression problems,
with scikit-learn compatible regressors and their respective hyperparameter search spaces.
"""

from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from skopt.space import Categorical, Integer, Real
from myml.optimization.search import HyperparameterSearchSpace, ModelSearchSpace
from myml.utils import get_mlp_hidden_layer_sizes


estimators = ModelSearchSpace()
""" Search space for regressors. """

estimators[LinearRegression()] = HyperparameterSearchSpace(
    fit_intercept=Categorical([True])
)

estimators[ElasticNet()] = HyperparameterSearchSpace(
    alpha=Real(0, 10),
    l1_ratio=Real(0, 1),
    max_iter=Integer(100, 100000, 'log-uniform')
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
    n_neighbors=Integer(3, 100),
    weights=Categorical(['uniform', 'distance']),
    p=Categorical([1, 2])
)

estimators[MLPRegressor()] = HyperparameterSearchSpace(
    hidden_layer_sizes=Categorical(get_mlp_hidden_layer_sizes([1,2,3], [16,32,64,128,256])),
    activation=Categorical(['relu', 'tanh']),
    alpha=Real(1e-6, 1e-1, 'log-uniform'),
    learning_rate_init=Real(1e-4, 1e-1, 'log-uniform')
)
