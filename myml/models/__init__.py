from myml.utils import ProblemType
from ._classification import estimators as classifiers
from ._regression import estimators as regressors

estimators = {
    ProblemType.classification: classifiers,
    ProblemType.regression: regressors
}
