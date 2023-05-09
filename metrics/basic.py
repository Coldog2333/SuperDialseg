import json
from typing import Optional, List, Union, Any
from prettytable import PrettyTable


class MetricBase:
    """
    评估指标基类
    """
    def __init__(self):
        self.metric_name = self.__class__.__name__

    @property
    def name(self):
        return self.metric_name

    def __call__(
        self,
        y_true: Union[List, Any],
        y_pred: Union[List, Any],
        *args, **kwargs
    ):
        """
        computation of a metric for a batch of data.
        TODO: should support on a batch of data. But currently, we only support on single data.
        param: y_true  -- ground truth [batch_size, *]
        param: y_pred  -- predictions  [batch_size, *]
        """
        raise NotImplementedError


class EvaluationBase:
    """
    借鉴huggingface的Evaluate库 (https://huggingface.co/docs/evaluate/index) 的设计思路。
    在每一个loop里收集下来y_true和y_pred，然后再统一计算所有的指标。
    每个指标的计算规则各有不同，需要分别实现。
    """
    def __init__(self, metrics: List[MetricBase]):
        self.metrics = metrics
        self.metric_name_dict = {metric.name: metric for metric in metrics}

        self.all_y_true = []
        self.all_y_pred = []
        self.result_dict = {k: 0. for k in self.metric_name_dict.keys()}
        self.result_dict.update({'#sample': 0, 'total_score': 0.})

    def add(
        self,
        y_true,
        y_pred
    ):
        """
        增加一个评估样本
        """
        assert (len(y_true) == len(y_pred))
        self.all_y_true.append(y_true)
        self.all_y_pred.append(y_pred)

    def add_batch(
        self,
        batch_y_true,
        batch_y_pred
    ):
        """
        添加一个batch的评估样本
        """
        assert (len(batch_y_true) == len(batch_y_pred))

        self.all_y_true.extend(batch_y_true)
        self.all_y_pred.extend(batch_y_pred)

    def _compute(
        self,
        y_true=None,
        y_pred=None
    ):
        result_dict = {k: 0. for k in self.metric_name_dict.keys()}
        result_dict.update({'#sample': len(y_true)})

        assert (len(y_true) == len(y_pred))

        for y_t, y_p in zip(y_true, y_pred):
            for metric in self.metrics:
                r = metric(y_true=y_t, y_pred=y_p)
                result_dict[metric.name] += r

        for k in self.result_dict.keys():
            if k not in ['#sample', 'total_score']:
                result_dict[k] /= result_dict['#sample']

        return result_dict

    def compute(self, y_true=None, y_pred=None, dump_file=None):
        self.reset()
        if y_true is None and y_pred is None:
            assert (len(self.all_y_true) != 0 or len(self.all_y_pred) != 0)
            outputs = self._compute(self.all_y_true, self.all_y_pred)

            # if implemented
            try:
                outputs['total_score'] = self.compute_total_score(outputs)
            except:
                pass

            if dump_file:
                self.__dump_result_to_file__(
                    all_y_true=self.all_y_true,
                    all_y_pred=self.all_y_pred,
                    dump_file=dump_file
                )
            self.all_y_true = []
            self.all_y_pred = []
            self.result_dict = outputs

            return outputs

        elif y_true is not None and y_pred is not None:
            return self._compute(y_true, y_pred)

        else:
            raise ValueError('y_true or y_pred is missing.')

    def show_performance(self, title='', auto_print=True):
        metric_names = list(self.metric_name_dict.keys())
        if 'total_score' in self.result_dict.keys():
            metric_names += ['total_score']

        tb = PrettyTable()
        tb.title = title
        tb.field_names = ['#Sample'] + metric_names
        tb.add_row([str(self.result_dict['#sample'])] + [str(round(self.result_dict[k], 4)) for k in metric_names])

        if auto_print:
            print('\n' + str(tb))

        return str(tb)

    def __dump_result_to_file__(self, all_y_true, all_y_pred, dump_file):
        with open(dump_file, 'w', encoding='utf-8') as writer:
            for i, (y_true, y_pred) in enumerate(zip(all_y_true, all_y_pred)):
                writer.write('=====Case_%s\n' % i)
                writer.write('Label:\t%s\n' % repr(y_true))
                writer.write('Pred:\t%s\n' % repr(y_pred))
                writer.write('=====\n')

    def compute_total_score(self):
        raise NotImplementedError

    def reset(self):
        self.result_dict = {k: 0. for k in self.metric_name_dict.keys()}
        self.result_dict.update({'#sample': 0, 'total_score': 0.})
