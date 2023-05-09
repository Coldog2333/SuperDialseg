import collections
import logging
import os
import re
import numpy as np
import torch
from numpy.linalg import norm
import nltk
import time
import torch.nn as nn
from tqdm import tqdm

from modeling_utils import BaseSegmenter
from models.embedding_texttiling.utils_embedding_texttiling import GloveTokenizer, SpaceTokenizer
from models.texttiling.utils_texttiling import depth_score_cal, cutoff_threshold
from secret_config import model_root_dir


class EmbeddingSegmenter(BaseSegmenter):
    def __init__(
        self,
        glove_path,
        cut_rate=0.8,
        text_preprocess_fn=lambda x: x,
    ):
        super().__init__()

        self.cut_rate = cut_rate
        self.text_preprocess_fn = text_preprocess_fn

        # load glove embedding
        ## vocab + embedding
        vocab = collections.OrderedDict()
        vocab['[PAD]'] = 0
        vocab['[UNK]'] = 1

        embeddings = [[0] * 300, np.random.normal(size=300).tolist()]

        logging.warning('loading word embedding...')

        n_word_counter = 0
        line_num = int(os.popen('wc -l {}'.format(glove_path)).read().split()[-2])
        with tqdm(total=line_num, desc='loading glove embedding') as pbar:
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.strip().split(' ')
                    word = values[0]
                    vocab[word] = len(vocab)

                    vector = [float(v) for v in values[1:]]
                    embeddings.append(vector)

                    n_word_counter += 1
                    # if n_word_counter > 50:
                    #     break

                    pbar.update(1)

        logging.warning('loaded.')

        self.vocab = vocab
        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=torch.tensor(embeddings, dtype=torch.float32),
            freeze=True,
            padding_idx=0,
        )

        self.tokenizer = GloveTokenizer(vocab=vocab)
        # self.tokenizer = SpaceTokenizer(vocab=vocab)

    def cosine_similarity_of_word_matrix(self, m1, m2):
        "m1, m2: Word Embedding Matrix with shape [b x N x d]"
        dot_product = torch.einsum('bnd,bmd->bnm', m1, m2)

        scale1 = torch.sqrt(torch.einsum('bnd,bnd->bn', m1, m1))
        scale2 = torch.sqrt(torch.einsum('bmd,bmd->bm', m2, m2))

        norm1 = scale1.unsqueeze(2).repeat(1, 1, m2.shape[1])
        norm2 = scale2.unsqueeze(1).repeat(1, m1.shape[1], 1)

        cos_similarity = dot_product / norm1 / norm2  # [b x N x M]
        # change nan to 0
        cos_similarity[cos_similarity != cos_similarity] = 0

        return cos_similarity

    def fast_similarity(
        self,
        s1,
        s2,
        attention_mask_1=None,
        attention_mask_2=None,
        heuristic='max'
    ) -> torch.Tensor:
        """
        :param s1:  [batch_size, seq_len_n, hidden_size]
        :param s2:  [batch_size, seq_len_m, hidden_size]
        :param heuristic: 'max' or 'aver'
        :return: similarity scores  [batch_size]
        """
        cos_similarity = self.cosine_similarity_of_word_matrix(m1=s1, m2=s2)    # [b x N x M]

        attention_mask = torch.einsum('bn,bm->bnm', attention_mask_1, attention_mask_2)  # [b x N x M]

        if heuristic == 'max':
            # apply maxpooling on cos_similarity to get the similarity by reducing the last two dimensions
            # apply attention_mask first
            if attention_mask is not None:
                cos_similarity = cos_similarity.masked_fill(attention_mask == 0, -1e9)
            # similarity = torch.nn.functional.max_pool2d(cos_similarity, kernel_size=cos_similarity.shape[1:]).squeeze()
            similarity = torch.max(cos_similarity, dim=-1).values

        elif heuristic == 'mean':
            # apply sum on cos_similarity to get the similarity by reducing the last two dimensions
            # apply attention_mask first
            if attention_mask is not None:
                cos_similarity = cos_similarity.masked_fill(attention_mask == 0, 0)

            similarity = cos_similarity.sum(dim=-1) / attention_mask.sum(-1)
            # similarity = torch.nn.functional.avg_pool2d(cos_similarity, kernel_size=cos_similarity.shape[1:]).squeeze()
        else:
            raise AssertionError('reduction should be [max] or [mean]')

        similarity = similarity * attention_mask_1

        similarity = similarity.sum(dim=-1) / attention_mask_1.sum(-1)

        return similarity

    def forward(self, inputs):
        utterances = [self.text_preprocess_fn(u) for u in inputs['utterances']]

        all_input_ids = []
        all_attention_mask = []
        for utterance in utterances:
            input_tokens = self.tokenizer.tokenize(utterance)

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            assert (len(input_ids) > 0 and len(input_tokens) > 0)

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor([1] * len(input_tokens), dtype=torch.long)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)

        all_input_ids = torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        all_attention_mask = torch.nn.utils.rnn.pad_sequence(all_attention_mask, batch_first=True, padding_value=0)

        embeddings = self.embeddings(all_input_ids)

        embeddings_a = embeddings[:-1, :, :]
        embeddings_b = embeddings[1:, :, :]

        attention_mask_a = all_attention_mask[:-1, :]   # [b, n]
        attention_mask_b = all_attention_mask[1:, :]    # [b, n]

        scores = self.fast_similarity(
            s1=embeddings_a,
            s2=embeddings_b,
            attention_mask_1=attention_mask_a,
            attention_mask_2=attention_mask_b,
            heuristic='max'
        )
        scores = scores.detach().numpy().tolist()

        depth_scores, mean, std = depth_score_cal(scores)
        threshold = cutoff_threshold(self.cut_rate, mean, std)

        # seg
        predictions = [0] * len(utterances)
        for i in range(len(depth_scores)):
            predictions[i] = 1 if depth_scores[i] > threshold else 0
        predictions[-1] = 1

        return predictions


if __name__ == '__main__':
    glove_path = os.path.join(model_root_dir, 'glove', 'glove.42B.300d.txt')
    segmenter = EmbeddingSegmenter(glove_path=glove_path)
