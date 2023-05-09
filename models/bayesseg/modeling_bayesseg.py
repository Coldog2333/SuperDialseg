import os
import json
from tqdm import tqdm

from modeling_utils import BaseSegmenter
from utils.data.load_dialseg import (
    load_segmentation_samples_from_json,
    load_segmentation_samples_from_txt
)
from secret_config import dialseg_code_root_dir, dataset_root_dir


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
        cmd = 'cat %s | %s/models/bayesseg/segment config/dp.config' % (dial_filename, dialseg_code_root_dir)
        raw_result = os.popen(cmd).read()
        segmentation_indices = json.loads(raw_result)

        predictions = [0] * line_num
        for index in segmentation_indices:
            predictions[index - 1] = 1
        predictions[-1] = 1

        return predictions

    # def transform_format_from_dialseg_to_bayesseg(self, segmentation_file, input_dir='data/seg-input'):
    #     if not os.path.exists(input_dir):
    #         os.mkdir(input_dir)
    #
    #     if segmentation_file.endswith('.json'):
    #         segmentation_samples = load_segmentation_samples_from_json(segmentation_file)
    #     else:
    #         segmentation_samples = load_segmentation_samples_from_txt(segmentation_file)
    #
    #     self.n_samples = len(segmentation_samples)
    #
    #     for i, sample in enumerate(segmentation_samples):
    #         with open(os.path.join(input_dir, 'dial_%03d.ref' % i), 'w', encoding='utf-8') as writer:
    #             writer.writelines([u + '\n' for u in sample.utterances])

    # def segmentation(self, segmentation_file, cache_dir='data/seg-input', dump=False, dump_file=''):
    #     # transform dialseg format to bayesseg format
    #     self.transform_format_from_dialseg_to_bayesseg(segmentation_file, input_dir=cache_dir)
    #
    #     results = []
    #     with tqdm(total=self.n_samples) as pbar:
    #         for i in range(self.n_samples):
    #             dial_filename = os.path.join(cache_dir, 'dial_%03d.ref' % i)
    #             line_num = int(os.popen('wc -l %s' % dial_filename).read().split(' ')[-2])
    #
    #             # run java
    #             cmd = 'cat %s | %s/bayesseg/segment config/dp.config' % (dial_filename, dialseg_code_root_dir)
    #             raw_result = os.popen(cmd).read()
    #             segmentation_indices = json.loads(raw_result)
    #
    #             predictions = [0] * line_num
    #             for index in segmentation_indices:
    #                 predictions[index - 1] = 1
    #
    #             results.append(predictions)
    #             pbar.update(1)
    #
    #     if dump:
    #         dataset_name = segmentation_file.split('/')[-2]  # e.g. 'super-dialseg'
    #         dump_file = "./segmentation_bayesseg_%s.json" % dataset_name if dump_file is None else dump_file
    #         with open(dump_file, 'w', encoding='utf-8') as writer:
    #             writer.write(json.dumps(results, ensure_ascii=False))
    #
    #     return results
    #
    # def forward(self, labels):
    #     if self.results is None:
    #         assert AssertionError("Please run self.segmentation(*args, **kwargs) first!")
    #
    #     predictions = self.results[self.counter]
    #     self.counter += 1
    #     return predictions
    #
    # def reset(self):
    #     self.counter = 0
