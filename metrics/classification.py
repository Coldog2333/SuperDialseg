from typing import Optional, List, Union, Any
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)

from metrics.basic import MetricBase, EvaluationBase


class Accuracy(MetricBase):
    def __init__(self):
        super(Accuracy, self).__init__()
        self.metric_name = 'accuracy'

    def __call__(
        self,
        y_true: List[int],
        y_pred: List[int],
        *args, **kwargs
    ):
        return accuracy_score(y_true=y_true, y_pred=y_pred)


class Precision(MetricBase):
    def __init__(self):
        super(Precision, self).__init__()
        self.metric_name = 'precision'

    def __call__(
        self,
        y_true: List[int],
        y_pred: List[int],
        *args, **kwargs
    ):
        return precision_score(y_true=y_true, y_pred=y_pred, average='binary', zero_division=0)


class Recall(MetricBase):
    def __init__(self):
        super(Recall, self).__init__()
        self.metric_name = 'recall'

    def __call__(
        self,
        y_true: List[int],
        y_pred: List[int],
        *args, **kwargs
    ):
        return recall_score(y_true=y_true, y_pred=y_pred, average='binary', zero_division=0)


class F1Score(MetricBase):
    def __init__(self, labels=None, average='binary'):
        super(F1Score, self).__init__()
        self.labels = labels     # None or [0,1]
        self.average = average   # None or 'macro'
        self.metric_name = 'f1'
        if self.average:
            self.metric_name += '(%s)' % self.average

    def __call__(
        self,
        y_true: List[int],
        y_pred: List[int],
        *args, **kwargs
    ):
        return f1_score(y_true=y_true, y_pred=y_pred, labels=self.labels, average=self.average, zero_division=0)


class ClassificationEvaluation(EvaluationBase):
    def __init__(self):
        super(ClassificationEvaluation, self).__init__(
            metrics=[Precision(),
                     Recall(),
                     F1Score(),
                     Accuracy()]
        )
