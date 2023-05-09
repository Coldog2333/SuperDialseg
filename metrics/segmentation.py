from typing import List, Union
from nltk.metrics import windowdiff as nltk_windowdiff
from nltk.metrics import pk as nltk_pk
import numpy as np

from metrics.basic import MetricBase, EvaluationBase
from metrics.classification import F1Score


class Pk(MetricBase):
    def __init__(self, window_size: Union[int, str] = 'auto'):
        super(Pk, self).__init__()
        self.window_size = window_size
        self.metric_name = 'pk'
        # self.metric_name += '-%s' % window_size

    def __call__(
        self,
        y_true: List[int],
        y_pred: List[int],
        *args, **kwargs
    ):
        if self.window_size == 'auto':
            n_segment = sum(y_true) + 1   # 00000100100100
            aver_length_of_segment = len(y_true) / n_segment / 2
            window_size = max(2, int(round(aver_length_of_segment)))
        else:
            window_size = self.window_size

        y_true = ''.join([str(v) for v in y_true])
        y_pred = ''.join([str(v) for v in y_pred])

        return nltk_pk(ref=y_true, hyp=y_pred, k=window_size)


class Windowdiff(MetricBase):
    def __init__(self, window_size: Union[int, str] = 'auto'):
        super(Windowdiff, self).__init__()
        self.window_size = window_size
        self.metric_name = 'windowdiff'
        # self.metric_name += '-%s' % window_size

    def __call__(
        self,
        y_true: List[int],
        y_pred: List[int],
        *args, **kwargs
    ):
        if self.window_size == 'auto':
            n_segment = sum(y_true) + 1
            aver_length_of_segment = len(y_true) / n_segment / 2
            window_size = max(2, int(round(aver_length_of_segment)))
        else:
            window_size = self.window_size

        y_true = ''.join([str(v) for v in y_true])
        y_pred = ''.join([str(v) for v in y_pred])

        return nltk_windowdiff(seg1=y_true, seg2=y_pred, k=window_size)


class MAE(MetricBase):
    def __init__(self):
        super(MAE, self).__init__()
        self.metric_name = 'mae'

    def __call__(
        self,
        y_true: List[int],
        y_pred: List[int],
        *args, **kwargs
    ):
        # return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        return np.abs(np.sum(y_true) - np.sum(y_pred))         # ref: TADAM-Evaluate p5


class SegmentationEvaluation(EvaluationBase):
    def __init__(self, window_size='auto'):
        super(SegmentationEvaluation, self).__init__(
            metrics=[Pk(window_size=window_size),
                     Windowdiff(window_size=window_size),
                     F1Score(),
                     F1Score(labels=[0, 1], average='macro'),
                     MAE()]
        )

    def compute_total_score(self, outputs=None):
        if outputs is None:
            outputs = self.result_dict

        return 0.5 * outputs['f1(binary)'] \
            + 0.25 * (1 - outputs['pk']) \
            + 0.25 * (1 - outputs['windowdiff'])
