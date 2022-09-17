from typing import Dict
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Categorical, Integer, Real, Dimension


models: Dict[BaseEstimator, Dict[str, Dimension]] = {
    LogisticRegression(): {
        'penalty': Categorical(['none', 'l2']),
        'max_iter': Integer(100, 1000)
    },
    RandomForestClassifier(): {
        'n_estimators': Integer(10, 200),
        'criterion': Categorical(['gini', 'entropy']),
        'min_samples_split': Integer(2, 200),
        'min_samples_leaf': Integer(1, 100)
    },
    GradientBoostingClassifier(): {
        'n_estimators': Integer(10, 400),
        'min_samples_split': Integer(2, 200),
        'min_samples_leaf': Integer(1, 100),
        'learning_rate': Real(1e-6, 1e-1, 'log-uniform')
    },
    AdaBoostClassifier(): {
        'n_estimators': Integer(10, 200),
    },
    KNeighborsClassifier(): {
        'n_neighbors': Integer(3, 50),
        'weights': Categorical(['uniform', 'distance'])
    }
}