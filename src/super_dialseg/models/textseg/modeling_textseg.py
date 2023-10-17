import math
import numpy as np
import torch
import torch.nn as nn
import gensim
from ...modeling_utils import BaseSegmenter
from .tokenization_textseg import TextsegTokenizer
from . import utils_textseg


class TextsegEmbedding(nn.Module):
    def __init__(self, word2vec):
        super(TextsegEmbedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=torch.tensor(word2vec.vectors, dtype=torch.float32))

    def forward(self, x):
        return self.embedding(x)


class SentenceEncodingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SentenceEncodingRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, inputs):
        packed_output, _ = self.lstm(inputs)
        padded_output, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # (max sentence len, batch, 256)
        # apply max pooling
        maxes, _ = torch.max(padded_output, dim=1)
        return maxes


class TextsegModel(nn.Module):
    def __init__(self, word2vec, hidden_size=256, num_layers=2):
        super(TextsegModel, self).__init__()
        self.embedding = TextsegEmbedding(word2vec=word2vec)
        self.sentence_encoder = SentenceEncodingRNN(input_size=300, hidden_size=256, num_layers=2)
        self.sentence_lstm = nn.LSTM(
            input_size=self.sentence_encoder.hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.,
            batch_first=True,
            bidirectional=True
        )

        # We have two labels
        self.h2s = nn.Linear(hidden_size * 2, 2)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs):
        batch = inputs['input_ids']

        word_embeddings = self.embedding(batch)

        sentences_per_doc = inputs['dialogue_length']
        lengths = inputs['utterance_length']

        packed_tensor = torch.nn.utils.rnn.pack_padded_sequence(input=word_embeddings, lengths=lengths, batch_first=True, enforce_sorted=False)
        encoded_sentences = self.sentence_encoder(packed_tensor)

        index = 0
        encoded_documents = []
        for sentences_count in sentences_per_doc:
            end_index = index + sentences_count
            encoded_documents.append(encoded_sentences[index:end_index, :])
            index = end_index

        docs_tensor = torch.nn.utils.rnn.pad_sequence(encoded_documents, batch_first=True, padding_value=0)

        packed_docs = torch.nn.utils.rnn.pack_padded_sequence(docs_tensor, sentences_per_doc, batch_first=True, enforce_sorted=False)
        # sentence_lstm_output, _ = self.sentence_lstm(packed_docs.to(device))
        sentence_lstm_output, _ = self.sentence_lstm(packed_docs)
        padded_x, _ = torch.nn.utils.rnn.pad_packed_sequence(sentence_lstm_output, batch_first=True)

        # -1 to remove last prediction
        doc_outputs = [padded_x[i, :doc_len - 1, :] for i, doc_len in enumerate(sentences_per_doc)]
        sentence_outputs = torch.cat(doc_outputs, dim=0)

        x = self.h2s(sentence_outputs)
        return x


class TextsegSegmenter(BaseSegmenter):
    def __init__(self, model_name_or_path=None, word2vecfile=None, threshold=0.4, device=None):
        super(TextsegSegmenter, self).__init__()
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vecfile, binary=True)
        self.model = TextsegModel(word2vec=word2vec, hidden_size=256, num_layers=2)
        self.model.load_state_dict(
            torch.load(open(model_name_or_path, 'rb'), map_location=torch.device(self.device)),
            strict=False
        )
        self.model.to(self.device)

        self.threshold = threshold

        self.tokenizer = TextsegTokenizer(word2vec=word2vec)

    def prepare_inputs(self, utterances):
        all_input_ids, all_utterance_length = [], []
        for utterance in utterances:
            tokens = self.tokenizer.tokenize(utterance, remove_special_tokens=True)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            assert (len(input_ids) >= 1), utterance

            utterance_length = len(input_ids)

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            all_input_ids.append(input_ids)
            all_utterance_length.append(utterance_length)

        # all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_ids = torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True)

        dialogue_length = [len(utterances)]

        return {
            'input_ids': all_input_ids,
            'utterance_length': all_utterance_length,
            'dialogue_length': dialogue_length
        }

    def forward(self, inputs):
        utterances = inputs['utterances']
        inputs = self.prepare_inputs(utterances)

        outputs = self.model(inputs)
        outputs = torch.softmax(outputs, dim=-1).detach().cpu().numpy()

        current_idx = 0
        for _, n in enumerate(inputs['dialogue_length']):
            to_idx = current_idx + n

            output = (outputs[current_idx: to_idx, 1] > self.threshold).astype(int)
            predictions = output.tolist()

            predictions += [1]

            current_idx = to_idx

        predictions[-1] = 0

        return predictions
