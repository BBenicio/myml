"""
Setup the search space for use of the `auto` module for classification problems.

The `estimators` object contains the search space for classification problems,
with scikit-learn compatible classifiers and their respective hyperparameter search spaces.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from skopt.space import Categorical, Integer, Real
from myml.optimization.search import HyperparameterSearchSpace, ModelSearchSpace
from myml.utils import get_mlp_hidden_layer_sizes


estimators = ModelSearchSpace()
""" Search space for classifiers. """

estimators[LogisticRegression()] = HyperparameterSearchSpace(
    penalty=Categorical(['none', 'l1', 'l2']),
    solver=Categorical(['saga']),
    max_iter=Integer(100, 100000, 'log-uniform')
)

estimators[GaussianNB()] = HyperparameterSearchSpace(
    priors=Categorical([None])
)

estimators[MultinomialNB()] = HyperparameterSearchSpace(
    fit_prior=Categorical([False, True])
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
    n_neighbors=Integer(3, 100),
    weights=Categorical(['uniform', 'distance']),
    p=Categorical([1, 2])
)

estimators[MLPClassifier()] = HyperparameterSearchSpace(
    hidden_layer_sizes=Categorical(get_mlp_hidden_layer_sizes([1,2,3], [16,32,64,128,256])),
    activation=Categorical(['relu', 'tanh']),
    alpha=Real(1e-6, 1e-1, 'log-uniform'),
    learning_rate_init=Real(1e-4, 1e-1, 'log-uniform')
)
