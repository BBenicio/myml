"""
Search spaces for each problem type.

The `estimators` object maps the search spaces for each problem type.
"""

from myml.utils import ProblemType
from ._classification import estimators as classifiers
from ._regression import estimators as regressors


estimators = {
    ProblemType.classification: classifiers,
    ProblemType.regression: regressors
}
""" Dictionary of the search spaces for each problem type supported. """
