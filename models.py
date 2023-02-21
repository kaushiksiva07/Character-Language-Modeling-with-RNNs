# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import collections

import utils


#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


def form_input(context, vocab_index) -> torch.Tensor:
    indices = []
    for letter in context:
        word_index = vocab_index.index_of(letter)
        if word_index == -1:
            word_index = 1
        indices.append(word_index)
    indices_tens = torch.LongTensor(indices)
    return indices_tens


class RNNClassifier(ConsonantVowelClassifier, nn.Module):
    def __init__(self, dict_size, input_size, hidden_size, dropout, output_size, rnn_type='lstm'):
        super(RNNClassifier, self).__init__()
        self.word_embedding = nn.Embedding(dict_size, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, dropout=dropout, batch_first=True)
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.g = nn.Tanh()
        self.V = nn.Linear(hidden_size, output_size)
        self.vocab_index = utils.Indexer()
        self.vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
        for char in self.vocab:
            self.vocab_index.add_and_get_index(char)

    def forward(self, x):

        embedded_input = self.word_embedding(x)

        # RNN expects a batch
        embedded_input = embedded_input.unsqueeze(0)
        # Note: the hidden state and cell state are 1x1xdim tensor: num layers * num directions x batch_size x dimensionality
        # So we need to unsqueeze to add these 1-dims.
        init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
                      torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float())
        output, (hidden_state, cell_state) = self.rnn(embedded_input, init_state)
        # Note: hidden_state is a 1x1xdim tensor: num layers * num directions x batch_size x dimensionality
        linear_hid = self.V(hidden_state)
        return linear_hid.squeeze(1)

    def predict(self, context):
        indices = []
        for letter in context:
            word_index = self.vocab_index.index_of(letter)
            if word_index == -1:
                word_index = 1
            indices.append(word_index)
        indices_tens = torch.LongTensor(indices)
        log_probs = self.forward(indices_tens)
        prediction = torch.argmax(log_probs)
        return prediction.item()


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    dict_size = 27
    input_size = 20
    # batch_size =
    hidden_size = 8
    output_size = 2
    num_epochs = 5
    initial_learning_rate = 0.0005
    dropout = 0

    rnn = RNNClassifier(dict_size, input_size, hidden_size, dropout, output_size, rnn_type='lstm')

    optimizer = optim.Adam(rnn.parameters(), initial_learning_rate)
    cross_loss = nn.CrossEntropyLoss(reduction='sum')

    for epoch in range(0, num_epochs):
        np.random.shuffle(train_cons_exs)
        np.random.shuffle(train_vowel_exs)
        total_loss = 0.0
        train_exs = train_cons_exs + train_vowel_exs
        np.random.shuffle(train_exs)
        for ex in train_exs:
            x = form_input(ex, vocab_index)
            y_onehot = torch.zeros(2)
            if ex in train_cons_exs:
                y_onehot.scatter_(0, torch.from_numpy(np.asarray(0, dtype=np.int64)), 1)
            else:
                y_onehot.scatter_(0, torch.from_numpy(np.asarray(1, dtype=np.int64)), 1)
            y_onehot = y_onehot.unsqueeze(0)
            rnn.zero_grad()
            log_probs = rnn.forward(x)
            loss = cross_loss(log_probs, y_onehot)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    return rnn


#####################
# MODELS FOR PART 2 #
#####################
def form_input2(context, vocab_index) -> torch.Tensor:

    indices = []
    for word in context:
        word_index = vocab_index.index_of(word)
        if word_index == -1:
            word_index = 1
        indices.append(word_index)
    indices_tens = torch.LongTensor(indices)
    return indices_tens


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel, nn.Module):
    def __init__(self, dict_size, input_size, hidden_size, dropout, output_size, rnn_type='lstm'):
        super(RNNLanguageModel, self).__init__()
        self.word_embedding = nn.Embedding(dict_size, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, dropout=dropout, batch_first=True)
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.g = nn.Tanh()
        self.V = nn.Linear(hidden_size, output_size)
        self.vocab_index = utils.Indexer()
        self.vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
        for char in self.vocab:
            self.vocab_index.add_and_get_index(char)

    def forward(self, x):
        embedded_input = self.word_embedding(x)

        # RNN expects a batch
        embedded_input = embedded_input.unsqueeze(0)
        # Note: the hidden state and cell state are 1x1xdim tensor: num layers * num directions x batch_size x dimensionality
        # So we need to unsqueeze to add these 1-dims.
        init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
                      torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float())
        output, (hidden_state, cell_state) = self.rnn(embedded_input, init_state)
        # Note: hidden_state is a 1x1xdim tensor: num layers * num directions x batch_size x dimensionality
        linear_out = self.V(output)
        return linear_out

    def get_next_char_log_probs(self, context):
        """
                Returns a log probability distribution over the next characters given a context.
                The log should be base e
                :param context: a single character to score
                :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
                """
        RNNLanguageModel.eval(self)
        indices = []
        for letter in context:
            letter_index = self.vocab_index.index_of(letter)
            indices.append(letter_index)
        indices_tens = torch.LongTensor(indices)
        log_probs = self.log_softmax(self.forward(indices_tens)).squeeze(0)
        y_probs = log_probs[-1].detach().numpy()
        return y_probs

    def get_log_prob_sequence(self, next_chars, context):
        """
                Scores a bunch of characters following context. That is, returns
                log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
                The log should be base e
                :param next_chars:
                :param context:
                :return: The float probability
                """

        RNNLanguageModel.eval(self)
        seq_indices = []
        sequence = context + next_chars
        for letter in sequence:
            letter_index = self.vocab_index.index_of(letter)
            seq_indices.append(letter_index)
        indices_tensor = torch.LongTensor(seq_indices)
        log_probs = self.log_softmax(self.forward(indices_tensor)).squeeze(0)

        seq_prob = 0
        prev_index = len(context) - 1
        for letter in next_chars:
            letter_index = self.vocab_index.index_of(letter)
            prob = log_probs[prev_index][letter_index]
            seq_prob += prob.item()
            prev_index += 1
        return seq_prob

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    dict_size = 27
    input_size = 20
    # batch_size =
    hidden_size = 20
    output_size = 27
    num_epochs = 10
    initial_learning_rate = 0.001
    dropout = 0
    chunk_size = 20

    lm = RNNLanguageModel(dict_size, input_size, hidden_size, dropout, output_size, rnn_type='lstm')

    optimizer = optim.Adam(lm.parameters(), initial_learning_rate)
    cross_loss = nn.CrossEntropyLoss(reduction='sum')

    for epoch in range(0, num_epochs):
        total_loss = 0.0
        text_chunks = [train_text[i:i + chunk_size] for i in range(0, len(train_text), chunk_size)]
        for gold in text_chunks:
            x_input = ' ' + gold[:-1]
            x = form_input2(x_input, vocab_index)
            y = form_input(gold, vocab_index)
            lm.zero_grad()
            log_probs = lm.forward(x).squeeze(0)
            loss = cross_loss(log_probs, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    return lm
