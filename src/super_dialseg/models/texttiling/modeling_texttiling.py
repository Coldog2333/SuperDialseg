import numpy as np
from nltk.tokenize import TextTilingTokenizer

from ...modeling_utils import BaseSegmenter
from ...models.texttiling.utils_texttiling import depth_score_cal, cutoff_threshold
from ...models.greedyseg.modeling_greedyseg import GreedySegmenter


class TexttilingSegmenter(BaseSegmenter):
    def __init__(
        self,
        w: int = 10,
        k: int = 6
    ):
        super().__init__()
        # 'zys-zh' w=20, k=6
        self.tt = TextTilingTokenizer(w=w, k=k)

    def forward(self, inputs):
        utterances = [u.strip('\n') for u in inputs['utterances']]
        document = '\n\n'.join(utterances)

        tiles = self.tt.tokenize(document)
        predictions = []
        for tile in tiles:
            lines = tile.strip().split('\n\n')
            if lines == ['']:  # bad case
                continue
            predictions.extend([0] * len(lines))
            predictions[-1] = 1

        predictions[-1] = 0

        return predictions


class TexttilingCLSSegmenter(GreedySegmenter):
    def __init__(
        self,
        backbone: str = 'bert-base-uncased',
        max_utterance_len: int = 50,
        cut_rate: float = -0.5,
    ):
        super().__init__(backbone=backbone, max_utterance_len=max_utterance_len)

        self.cut_rate = cut_rate

    def generate_vectors(self, sentences):
        inputs = self.tokenizer(
            sentences,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_utterance_len,
            return_tensors='pt'
        )

        outputs = self.model(
            input_ids=inputs['input_ids'].to(self.device),
            token_type_ids=inputs['token_type_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            return_dict=True
        )

        last_hidden_state = outputs['last_hidden_state']
        cls_vectors = last_hidden_state[:, 0, :]
        return cls_vectors.cpu().detach().numpy()

    def forward(self, inputs):
        utterances = inputs['utterances']
        vectors = self.generate_vectors(utterances)

        paired_vector_1, paired_vector_2 = [], []
        for i in range(len(vectors) - 1):
            paired_vector_1.append(vectors[i])
            paired_vector_2.append(vectors[i+1])
        paired_vector_1 = np.array(paired_vector_1)
        paired_vector_2 = np.array(paired_vector_2)

        scores = np.sum(paired_vector_1 * paired_vector_2, axis=1) \
                 / (np.linalg.norm(paired_vector_1, axis=1) *
                    np.linalg.norm(paired_vector_2, axis=1))

        depth_scores, _, _ = depth_score_cal(scores)

        ##### top-k startegy
        # pick_num = 4
        # boundary_indice = np.argsort(np.array(depth_scores))[-pick_num:]
        #
        # predictions = [0] * (len(depth_scores) + 1)
        # for i in boundary_indice:
        #     predictions[i] = 1
        # predictions[-1] = 1

        ##### by threshold
        threshold = cutoff_threshold(cut_rate=self.cut_rate,
                                     mean=np.mean(depth_scores),
                                     std=np.std(depth_scores))

        predictions = []
        for depth_score in depth_scores:
            if depth_score > threshold:
                predictions.append(1)
            else:
                predictions.append(0)
        predictions.append(1)

        predictions[-1] = 0

        return predictions


if __name__ == '__main__':
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    depth_scores, mean, std = depth_score_cal(scores)
    print(depth_scores)
    print(mean)
    print(std)
    print(cutoff_threshold(0.5, mean, std))
