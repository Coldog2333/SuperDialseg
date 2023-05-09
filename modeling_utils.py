from metrics.segmentation import SegmentationEvaluation


class BaseSegmenter:
    def __init__(self):
        self.evaluation = SegmentationEvaluation(window_size='auto')

    def batch_forward(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        "labels: e.g. [0, 0, 1, 0, 1, 0, 1]"
        raise NotImplementedError
