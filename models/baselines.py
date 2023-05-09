import json
import random
from modeling_utils import BaseSegmenter


class RandomSegmenter(BaseSegmenter):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, ignore_index=-100):
        # labels: e.g. [0, 0, 1, 0, 1, 0, 1]
        labels = [label for label in inputs['labels'] if label != ignore_index]

        k = len(labels)
        b = random.randint(0, k - 1)

        predictions = []

        for i in range(k):
            if random.random() < b/k:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions


class EvenSegmenter(BaseSegmenter):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, ignore_index=-100):
        # labels: e.g. [0, 0, 1, 0, 1, 0, 1]
        labels = [label for label in inputs['labels'] if label != ignore_index]

        k = len(labels)
        b = random.randint(0, k - 1)

        if b == 0:
            return [0] * k
        else:
            predictions = []
            interval = int(k / b)
            for i in range(b):
                predictions.extend([0] * interval)
                predictions[-1] = 1
            predictions += [0] * (k - len(predictions))

            return predictions


class ResultSegmenter(BaseSegmenter):
    """
        Do evaluation with json file.
        Note: Cannot shuffle the test samples because of the order of the results
    """
    def __init__(self, result_filename):
        super().__init__()

        self.results = json.load(open(result_filename))
        self.counter = 0

    def forward(self, inputs=None):
        predictions = self.results[self.counter]
        self.counter += 1
        return predictions

    def reset(self):
        self.counter = 0
