from typing import Dict
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from skopt.space import Categorical, Integer, Real, Dimension


models: Dict[BaseEstimator, Dict[str, Dimension]] = {
    Ridge(): {
        'alpha': Real(0, 2),
        'max_iter': Integer(100, 1000)
    },
    RandomForestRegressor(): {
        'n_estimators': Integer(10, 200),
        'criterion': Categorical(['squared_error', 'friedman_mse', 'absolute_error']),
        'min_samples_split': Integer(2, 200),
        'min_samples_leaf': Integer(1, 100)
    },
    GradientBoostingRegressor(): {
        'n_estimators': Integer(10, 400),
        'min_samples_split': Integer(2, 200),
        'min_samples_leaf': Integer(1, 100),
        'learning_rate': Real(1e-6, 1e-1, 'log-uniform')
    },
    AdaBoostRegressor(): {
        'n_estimators': Integer(10, 200),
    },
    KNeighborsRegressor(): {
        'n_neighbors': Integer(3, 50),
        'weights': Categorical(['uniform', 'distance'])
    }
}