import collections

import nltk
import numpy as np
import os
from typing import Union, List

from tqdm import tqdm


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def get_vocab_file_from_glove_embedding(glove_path):
    glove_dir, glove_filename = os.path.split(glove_path)
    vocab_file = os.path.join(glove_dir, glove_filename.replace('.txt', '.vocab'))

    # load glove embedding
    line_num = int(os.popen('wc -l {}'.format(glove_path)).read().split()[-2])
    vocab = ['[PAD]', '[UNK]']
    with tqdm(total=line_num, desc='loading glove embedding') as pbar:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split(' ')
                word = values[0]
                vocab.append(word)
                pbar.update(1)

    # write vocab file
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(vocab))
        f.write('\n')

    return vocab


class BaseTokenizer:
    def __init__(
        self,
        vocab: collections.OrderedDict = None,
        do_lower_case=True,
        unk_token="[UNK]",
        pad_token="[PAD]",
        **kwargs,
    ):
        self.do_lower_case = do_lower_case
        self.unk_token = unk_token
        self.pad_token = pad_token

        self.vocab = vocab
        if self.vocab:
            self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        tokens = []
        for index in ids:
            index = int(index)

            if skip_special_tokens:
                raise NotImplementedError('[TODO] skip_special_tokens is not implemented yet.')

            tokens.append(self._convert_id_to_token(index))

        return tokens

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_token_id(self):
        return self.vocab.get(self.pad_token)

    @property
    def unk_token_id(self):
        return self.vocab.get(self.unk_token)

class GloveTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab: collections.OrderedDict = None,
        vocab_file: Union[str, os.PathLike] = None,
        do_lower_case=True,
        unk_token="[UNK]",
        pad_token="[PAD]",
        **kwargs,
    ):
        super(GloveTokenizer, self).__init__(
            vocab=vocab,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

        self.vocab = vocab if vocab is not None else load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

    def tokenize(self, sentence: str):
        tokens = nltk.tokenize.word_tokenize(sentence)

        if self.do_lower_case:
            tokens = [token.lower() for token in tokens]

        return tokens


class SpaceTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab: collections.OrderedDict = None,
        do_lower_case=True,
        unk_token="[UNK]",
        pad_token="[PAD]",
        **kwargs,
    ):
        super(SpaceTokenizer, self).__init__(
            vocab=vocab,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

    def tokenize(self, sentence: str):
        tokens = sentence.split(' ')

        # remove empty tokens
        tokens = [token for token in tokens if token != '']

        if self.do_lower_case:
            tokens = [token.lower() for token in tokens]

        return tokens


if __name__ == '__main__':
    from secret_config import model_root_dir
    #
    # glove_path = os.path.join(model_root_dir, 'glove', 'glove.42B.300d.txt')
    # vocab = get_vocab_file_from_glove_embedding(glove_path)
    vocab_file = os.path.join(model_root_dir, 'glove', 'glove.42B.300d.vocab')
    tokenizer = GloveTokenizer(
        vocab_file=vocab_file,
        do_lower_case=True,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )

    sentence = 'Hello, world!'
    tokens = tokenizer.tokenize(sentence)
    print(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(input_ids)
