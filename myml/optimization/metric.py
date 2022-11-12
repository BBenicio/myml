"""
Metrics used to optimize and evaluate performance.

Multiple scikit-learn compatible scorers and metrics can be used for
evaluating the machine learning models trained. This module interfaces
with the scikit-learn metrics to help comparison and separation of use.
"""

from enum import Enum
from typing import Iterable, Literal, Optional, Callable, Any
from myml.utils import ProblemType


class MetricConfig:
    """
    Configuration of a metric.

    Parameters
    ----------
    sklearn_name : str
        Name of the metric on the sckit-learn library.
    type : MetricConfig.Type
        Whether the metric is maximizable or minimizable.
    used_for : ProblemType
        Which problem type is the metric suitable for.
    """

    Type = Literal['maximize', 'minimize']
    """ Type for deciding the kind of optimization problem for the metric. """

    def __init__(self, sklearn_name: str, type: "MetricConfig.Type", used_for: ProblemType) -> None:
        self.sklearn_name = sklearn_name
        self.type = type
        self.used_for = used_for

    @property
    def is_maximize(self) -> bool:
        """ Whether this metric should be maximized. """
        return self.type == 'maximize'

    @property
    def is_minimize(self) -> bool:
        """ Whether this metric should be minimized. """
        return self.type == 'minimize'


class Metric(Enum):
    """
    Metric for evaluating performance of machine learning models.
    
    See Also
    --------
    sklearn.metrics : Module that provides the metrics used.
    """
    # Classification
    accuracy = MetricConfig('accuracy', 'maximize', ProblemType.classification)
    f1 = MetricConfig('f1', 'maximize', ProblemType.classification)
    f1_micro = MetricConfig('f1_micro', 'maximize', ProblemType.classification)
    f1_macro = MetricConfig('f1_macro', 'maximize', ProblemType.classification)
    roc_auc = MetricConfig('roc_auc', 'maximize', ProblemType.classification)
    log_loss = MetricConfig('neg_log_loss', 'maximize', ProblemType.classification)

    # Regression
    max_error = MetricConfig('max_error', 'minimize', ProblemType.regression)
    mae = MetricConfig('neg_mean_absolute_error', 'maximize', ProblemType.regression)
    mse = MetricConfig('neg_mean_squared_error', 'maximize', ProblemType.regression)
    r2 = MetricConfig('r2', 'maximize', ProblemType.regression)

    # Type hinting the value
    @property
    def value(self) -> MetricConfig:
        return super().value

def _get_metric_config(metric: Metric | MetricConfig) -> MetricConfig:
    """
    Get a `MetricConfig` object from a metric if needed.

    Parameters
    ----------
    metric : Metric or MetricConfig
        The object to use for retrieving the MetricConfig.
    
    Returns
    -------
    MetricConfig
        The MetricConfig object retrieved.
    """
    return metric.value if isinstance(metric, Metric) else metric

def is_better(metric: Metric | MetricConfig, value: float, other: float) -> bool:
    """
    Check if the value is better than the other for the given metric.

    Considering the metric given, especifically if we should be maximizing
    or minimizing the values, see if `value` is better than `other`.

    Parameters
    ----------
    metric : Metric or MetricConfig
        The metric to consider for comparison.
    value : float
        The left hand of the "greater than" comparison.
    other : float
        The right hand of the "greater than" comparison.
    
    Returns
    -------
    bool
        Whether the `value` is better than `other` for the metric given.
    """
    mtc: MetricConfig = _get_metric_config(metric)
    return (other is None) or (mtc.is_maximize and value > other) or (mtc.is_minimize and value < other)

def translate_metric(metric: Metric | MetricConfig, value: float) -> float:
    """
    Translate the metric from/to a minimization problem.

    Parameters
    ----------
    metric : Metric or MetricConfig
        The metric to translate.
    value : float
        The value to translate.

    Returns
    -------
    float:
        The translated value for the metric from/to a minimization problem.
    """
    mtc: MetricConfig = _get_metric_config(metric)
    if mtc.is_maximize:
        return -value
    elif mtc.is_minimize:
        return value
    raise ValueError()

def sort_by_metric(to_sort: Iterable, metric: Metric | MetricConfig, key: Optional[Callable[[Any], Any]] = None) -> Iterable:
    """
    Sort a collection of items by the metric value for a given metric.

    According to the given metric, sort the collection so that the best
    elements occupy the first positions, and the worst are on the last
    positions.

    Parameters
    ----------
    to_sort : Iterable
        Collection of items to sort.
    metric: Metric or MetricConfig
        Metric to consider when sorting the collection.
    key: Callable, optional
        Function to call for retrieving the value of the metric from the items.

    Returns
    -------
    Iterable
        Sorted collection of items.

    See Also
    --------
    sorted : Sort lists given an optional `key` function.
    """
    mtc: MetricConfig = _get_metric_config(metric)
    return sorted(to_sort, key=key, reverse=mtc.is_maximize)
    