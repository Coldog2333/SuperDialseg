from .metrics.segmentation import SegmentationEvaluation


class BaseSegmenter:
    def __init__(self):
        self.evaluation = SegmentationEvaluation(window_size='auto')

    def batch_forward(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        "labels: e.g. [0, 0, 1, 0, 1, 0, 1]"
        raise NotImplementedError

    def __call__(self, utterances, auto_print=True, *args, **kwargs):
        inputs = {'utterances': utterances}
        predictions = self.forward(inputs, *args, **kwargs)

        predictions[-1] = 0

        print_str = ''
        for pred, utterance in zip(predictions, utterances):
            print_str += utterance + '\n'
            if pred == 1:
                print_str += '-' * 20 + '\n'

        if auto_print:
            print(print_str)

        return predictions, print_str
