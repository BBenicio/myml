from enum import Enum
from typing import Literal
from myml.utils import ProblemType


class MetricConfig:
    Type = Literal['maximize', 'minimize']

    def __init__(self, sklearn_name: str, type: "MetricConfig.Type", used_for: ProblemType) -> None:
        self.sklearn_name = sklearn_name
        self.type = type
        self.used_for = used_for

    @property
    def is_maximize(self) -> bool:
        return self.type == 'maximize'

    @property
    def is_minimize(self) -> bool:
        return self.type == 'minimize'


class Metric(Enum):
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
    return metric.value if isinstance(metric, Metric) else metric

def is_better(metric: Metric | MetricConfig, value: float, other: float):
    mtc: MetricConfig = _get_metric_config(metric)
    return (other is None) or (mtc.is_maximize and value > other) or (mtc.is_minimize and value < other)

def translate_metric(metric: Metric | MetricConfig, value: float) -> float:
    mtc: MetricConfig = _get_metric_config(metric)
    if mtc.is_maximize:
        return -value
    elif mtc.is_minimize:
        return value
    raise ValueError()
