import os
import json

from ...modeling_utils import BaseSegmenter
from .__init__ import CONFIGPATH

parent_dir = os.path.dirname(os.path.abspath(__file__))
dialseg_code_root_dir = os.path.join(parent_dir, '../..')


class BayesSegmenter(BaseSegmenter):
    def __init__(self):
        super().__init__()
        self.results = None
        self.counter = 0

        self.n_samples = 0

    def forward(self, inputs):
        utterances = inputs['utterances']

        cache_input_dir = os.path.join(dialseg_code_root_dir, '.cache/bayesseg/seg-input')
        if not os.path.exists(cache_input_dir):
            os.makedirs(cache_input_dir)

        dial_filename = os.path.join(cache_input_dir, 'dial.ref')
        with open(dial_filename, 'w', encoding='utf-8') as writer:
            writer.writelines([u + '\n' for u in utterances])

        line_num = int(os.popen('wc -l %s' % dial_filename).read().split(' ')[-2])
        # run java
        cmd = 'cat %s | %s/segment %s' % (dial_filename, parent_dir, CONFIGPATH)
        raw_result = os.popen(cmd).read()
        segmentation_indices = json.loads(raw_result)

        predictions = [0] * line_num
        for index in segmentation_indices:
            predictions[index - 1] = 1
        predictions[-1] = 1

        return predictions
