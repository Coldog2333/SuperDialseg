import os
import argparse
from multiprocessing import cpu_count
import re

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, SequentialSampler

from modeling_utils import BaseSegmenter
from models.baselines import (
    RandomSegmenter,
    EvenSegmenter,
    ResultSegmenter
)
from models.texttiling.modeling_texttiling import (
    TexttilingSegmenter,
    TextTilingCLSSegmenter
)
from models.bayesseg.modeling_bayesseg import BayesSegmenter
from models.csm.modeling_csm import (
    TexttilingNSPSegmenter,
    CSMSegmenter
)
from models.graphseg.modeling_graphseg import GraphsegSegmenter
from models.embedding_texttiling.modeling_embedding_texttiling import EmbeddingSegmenter
from models.greedyseg.modeling_greedyseg import GreedySegmenter
from utils.data.load_dialseg import DialsegDataset
from utils.data.data_collator import (
    DataCollatorForSegmentationLabels,
    DataCollatorPlainText
)

from secret_config import dialseg_code_root_dir, model_root_dir


class ReproducePipeline:
    def __init__(self, segmenter: BaseSegmenter):
        self.segmenter = segmenter

    def prepare_data(
        self,
        dataset_name: str = None,
        segmentation_file: str = None,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn=DataCollatorForSegmentationLabels(),
    ):
        if segmentation_file is None:
            segmentation_file = os.path.join(
                dialseg_code_root_dir,
                '.cache/datasets',
                f'{dataset_name}/segmentation_file_test.json'
            )

        test_dataset = DialsegDataset(segmentation_file=segmentation_file)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=batch_size,  # test_batch_size
            num_workers=num_workers,
            persistent_workers=True,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn
        )

        return test_dataloader

    def easy_evaluate(
        self,
        dataset_name: str = None,
        segmentation_file: str = None,
        batch_size: int = 1,
        num_workers: int = 0,
        show_performance=True,
        ignore_index=-100,
        collate_fn=DataCollatorForSegmentationLabels(),
    ):
        test_dataloader = self.prepare_data(
            dataset_name=dataset_name,
            segmentation_file=segmentation_file,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        with tqdm(total=len(test_dataloader)) as pbar:
            for i, batch in enumerate(test_dataloader):
                for j, labels in enumerate(batch['labels']):

                    # take the j-th data from batch
                    inputs = {k: v[j] for k, v in batch.items()}

                    # try to convert to list
                    try:
                        labels = labels.cpu().numpy().tolist()
                    except:
                        pass

                    segmentation_labels = []
                    for label in labels:
                        if label != ignore_index:
                            segmentation_labels.append(label)

                    segmentation_predictions = self.segmenter.forward(inputs)

                    # set the last prediction/label to 0, so that the value will be in the range of [0, 1]
                    segmentation_labels[-1] = 0
                    segmentation_predictions[-1] = 0

                    self.segmenter.evaluation.add(segmentation_labels, segmentation_predictions)
                pbar.update(1)

        results = self.segmenter.evaluation.compute()
        if show_performance:
            self.segmenter.evaluation.show_performance()

        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")

    # Testing
    parser.add_argument("--dataset_name", type=str, default="super_dialseg",
                        help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--segmentation_file", type=str, default=None, help="The path of the segmentation file.")
    parser.add_argument("--test_batch_size", type=int, default=128, help="Batch size for testing.")
    parser.add_argument("--num_workers", type=int, default=cpu_count(), help="Number of workers for the data loader.")
    parser.add_argument("--n_try", type=int, default=1, help="Number of tries for the evaluation.")
    # Model
    parser.add_argument('--model', default='random', type=str)
    parser.add_argument("--max_utterance_len", default=50, type=int)
    parser.add_argument("--cut_rate", default=0.5, type=float)
    # Environment
    parser.add_argument('--result_filename', type=str, default=None, help='Filename containing labels.')
    parser.add_argument('--glove_path', type=str, default=None, help='Path to GloVe embedding file.')
    parser.add_argument('--model_name_or_path', type=str, default=None, help='Path to pretrained model or model identifier.')
    parser.add_argument("--cache_dir", type=str, default=None, help="")

    # graphseg
    parser.add_argument("--jar_path", default='models/graphseg/binary/graphseg.jar', type=str, help='jar for graphseg')
    parser.add_argument("--relatedness_threshold",
                        default=0.25,
                        type=float,
                        help="""It is the value of the relatedness treshold (decimal number) to be used in 
                                the construction of the relatedness graph: larger values will give large number of 
                                smalled segments, whereas the smaller treshold values will provide a smaller number 
                                of coarse segments;""")
    parser.add_argument("--min_seg_size",
                        default=3,
                        type=int,
                        help="""It defines the minimal segment size m (in number of sentences). 
                                This means that GraphSeg will not produce segments containing less than m sentences. """)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    supporting_models = [
        'random', 'even',                                   # random baselines
        'texttiling', 'bayesseg',                                      # traditional algorithms
        'greedyseg', 'texttiling_cls', 'texttiling_glove',  # neural unsupervised models
        'texttiling_nsp', 'csm',
        'result'
    ]
    assert (args.model in supporting_models), f'Invalid model name: {args.model}'


    # model selection
    if args.model == 'random':
        segmenter = RandomSegmenter()
        collate_fn = DataCollatorForSegmentationLabels()

    elif args.model == 'even':
        segmenter = EvenSegmenter()
        collate_fn = DataCollatorForSegmentationLabels()

    elif args.model == 'texttiling':
        segmenter = TexttilingSegmenter(w=10, k=6)
        collate_fn = DataCollatorPlainText()

    # graphseg cannot work effectively by segmenting one-by-one
    # elif args.model == 'graphseg':
    #     segmenter = GraphsegSegmenter(
    #         jar_path=args.jar_path,
    #         relatedness_threshold=args.relatedness_threshold,
    #         min_seg_size=args.min_seg_size
    #     )
    #     collate_fn = DataCollatorPlainText()

    elif args.model == 'bayesseg':
        segmenter = BayesSegmenter()
        collate_fn = DataCollatorPlainText()

    elif args.model == 'greedyseg':
        segmenter = GreedySegmenter(backbone='bert-base-uncased', max_utterance_len=args.max_utterance_len)
        collate_fn = DataCollatorPlainText()

    elif args.model == 'texttiling_cls':
        segmenter = TextTilingCLSSegmenter(
            backbone='bert-base-uncased',
            max_utterance_len=args.max_utterance_len,
            cut_rate=args.cut_rate
        )
        collate_fn = DataCollatorPlainText()

    elif args.model == 'texttiling_nsp':
        segmenter = TexttilingNSPSegmenter(
            backbone='bert-base-uncased',
            max_utterance_len=args.max_utterance_len,
            cut_rate=args.cut_rate
        )
        segmenter.to(device)
        collate_fn = DataCollatorPlainText()

    elif args.model == 'csm':
        segmenter = CSMSegmenter(
            backbone='bert-base-uncased',
            max_utterance_len=args.max_utterance_len,
            cut_rate=args.cut_rate
        )

        if args.model_name_or_path is None:
            args.model_name_or_path = os.path.join('.cache/model_zoo', 'csm', 'csm-dailydial.pkl')

        segmenter.load_state_dict(ckpt_path=args.model_name_or_path)
        segmenter.to(device)
        collate_fn = DataCollatorPlainText()

    elif args.model == 'texttiling_glove':
        if args.glove_path is None:
            args.glove_path = os.path.join(model_root_dir, 'glove', 'glove.42B.300d.txt')

        segmenter = EmbeddingSegmenter(
            glove_path=args.glove_path,
            cut_rate=args.cut_rate,
            text_preprocess_fn=lambda x: x
            # text_preprocess_fn=lambda x: re.sub(r'[^\w\s]', '', x.strip().lower())
        )
        collate_fn = DataCollatorPlainText()

    else:
        segmenter = ResultSegmenter(args.result_filename)
        collate_fn = DataCollatorForSegmentationLabels()


    pipeline = ReproducePipeline(segmenter)


    # evaluation
    dataset_names = args.dataset_name.split(',')
    for dataset_name in dataset_names:
        aver_result = {'pk': 0., 'windowdiff': 0., 'f1(binary)': 0., 'f1(macro)': 0., 'mae': 0., 'total_score': 0., '#sample': 0.}
        print(f'Evaluating {segmenter.__class__.__name__} on {dataset_name}')
        with tqdm(total=args.n_try) as pbar:
            for _ in range(args.n_try):
                results = pipeline.easy_evaluate(
                    dataset_name=dataset_name,
                    segmentation_file=None,
                    batch_size=args.test_batch_size,
                    num_workers=args.num_workers,
                    show_performance=False,
                    ignore_index=-100,
                    collate_fn=collate_fn,
                )
                for k, v in results.items():
                    aver_result[k] += v / args.n_try
                pbar.update(1)

        aver_result['#sample'] = round(aver_result['#sample'])
        pipeline.segmenter.evaluation.result_dict = aver_result
        pipeline.segmenter.evaluation.show_performance(title=f"{args.model}\'s Performances on {dataset_name}")

