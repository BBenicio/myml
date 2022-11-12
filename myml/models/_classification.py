"""
Setup the search space for use of the `auto` module for classification problems.

The `estimators` object contains the search space for classification problems,
with scikit-learn compatible classifiers and their respective hyperparameter search spaces.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Categorical, Integer, Real
from myml.optimization.search import HyperparameterSearchSpace, ModelSearchSpace


estimators = ModelSearchSpace()
""" Search space for classifiers. """

estimators[LogisticRegression()] = HyperparameterSearchSpace(
    penalty=Categorical(['none', 'l2']),
    max_iter=Integer(100, 1000)
)

estimators[RandomForestClassifier()] = HyperparameterSearchSpace(
    n_estimators=Integer(10, 200),
    criterion=Categorical(['gini', 'entropy']),
    min_samples_split=Integer(2, 200),
    min_samples_leaf=Integer(1, 100)
)

estimators[GradientBoostingClassifier()] = HyperparameterSearchSpace(
    n_estimators=Integer(10, 400),
    min_samples_split=Integer(2, 200),
    min_samples_leaf=Integer(1, 100),
    learning_rate=Real(1e-6, 1e-1, 'log-uniform')
)

estimators[AdaBoostClassifier()] = HyperparameterSearchSpace(
    n_estimators=Integer(10, 200),
)

estimators[KNeighborsClassifier()] = HyperparameterSearchSpace(
    n_neighbors=Integer(3, 50),
    weights=Categorical(['uniform', 'distance'])
)
