import numpy as np
from nltk.tokenize import RegexpTokenizer

missing_stop_words = set(['of', 'a', 'and', 'to'])

class TextsegTokenizer:
    def __init__(self, word2vec):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.word2vec = word2vec   # default: not loaded, output random  np.random.randn(1, 300)

    def tokenize(self, sentence, remove_missing_emb_words=False, remove_special_tokens=True):
        if remove_special_tokens:
            for token in ["***LIST***", "***formula***", "***codice***"]:
                # Can't do on sentence words because tokenizer delete '***' of tokens.
                sentence = sentence.replace(token, "")

        tokenized_tokens = self.tokenizer.tokenize(sentence)
        tokens = []
        for token in tokenized_tokens:
            # check whether token is in word2vec
            try:
                _ = self.word2vec.key_to_index[token]
                tokens.append(token)
            except Exception as e:
                tokens.append('UNK')

        if remove_missing_emb_words:
            tokens = [w for w in tokens if w not in missing_stop_words]

        if tokens == []:
            tokens = ['UNK']

        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.word2vec.key_to_index[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.word2vec.index_to_key[id] for id in ids]