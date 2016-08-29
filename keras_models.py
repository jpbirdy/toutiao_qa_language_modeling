from __future__ import print_function

from abc import abstractmethod

from keras.engine import Input
from keras.layers import merge, Embedding, Dropout, Convolution1D, Lambda, LSTM, Dense, TimeDistributed, constraints
from keras import backend as K
from keras.models import Model

import numpy as np

from attention_lstm import AttentionLSTM


class LanguageModel:
  def __init__(self, config):
    self.question = Input(shape=(config['question_len'],), dtype='int32', name='question_base')
    self.answer_good = Input(shape=(config['answer_len'],), dtype='int32', name='answer_good_base')
    # self.answer_bad = Input(shape=(config['answer_len'],), dtype='int32', name='answer_bad_base')

    self.config = config
    self.model_params = config.get('model_params', dict())
    self.similarity_params = config.get('similarity_params', dict())

    # initialize a bunch of variables that will be set later
    self._models = None
    self._similarities = None
    self._answer = None
    self._qa_model = None

    self.training_model = None
    self.prediction_model = None

  def get_answer(self):
    # if self._answer is None:
    #   self._answer = Input(shape=(self.config['answer_len'],), dtype='int32', name='answer')
    # return self._answer
    return self.answer_good

  @abstractmethod
  def build(self):
    return

  def get_similarity(self):
    ''' Specify similarity in configuration under 'similarity_params' -> 'mode'
    If a parameter is needed for the model, specify it in 'similarity_params'

    Example configuration:

    config = {
        ... other parameters ...
        'similarity_params': {
            'mode': 'gesd',
            'gamma': 1,
            'c': 1,
        }
    }

    cosine: dot(a, b) / sqrt(dot(a, a) * dot(b, b))
    polynomial: (gamma * dot(a, b) + c) ^ d
    sigmoid: tanh(gamma * dot(a, b) + c)
    rbf: exp(-gamma * l2_norm(a-b) ^ 2)
    euclidean: 1 / (1 + l2_norm(a - b))
    exponential: exp(-gamma * l2_norm(a - b))
    gesd: euclidean * sigmoid
    aesd: (euclidean + sigmoid) / 2
    '''

    params = self.similarity_params
    similarity = params['mode']

    dot = lambda a, b: K.batch_dot(a, b, axes=1)
    l2_norm = lambda a, b: K.sqrt(K.sum((a - b) ** 2, axis=1, keepdims=True))

    if similarity == 'cosine':
      return lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())
    elif similarity == 'polynomial':
      return lambda x: (params['gamma'] * dot(x[0], x[1]) + params['c']) ** params['d']
    elif similarity == 'sigmoid':
      return lambda x: K.tanh(params['gamma'] * dot(x[0], x[1]) + params['c'])
    elif similarity == 'rbf':
      return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]) ** 2)
    elif similarity == 'euclidean':
      return lambda x: 1 / (1 + l2_norm(x[0], x[1]))
    elif similarity == 'exponential':
      return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]))
    elif similarity == 'gesd':
      euclidean = lambda x: 1 / (1 + l2_norm(x[0], x[1]))
      sigmoid = lambda x: 1 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
      return lambda x: euclidean(x) * sigmoid(x)
    elif similarity == 'aesd':
      euclidean = lambda x: 0.5 / (1 + l2_norm(x[0], x[1]))
      sigmoid = lambda x: 0.5 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
      return lambda x: euclidean(x) + sigmoid(x)
    else:
      raise Exception('Invalid similarity: {}'.format(similarity))

  def get_qa_model(self):
    if self._models is None:
      self._models = self.build()

    if self._qa_model is None:
      question_output, answer_output = self._models
      dropout = Dropout(self.similarity_params.get('similarity_dropout', 0.2))
      similarity = lambda x: K.expand_dims(self.get_similarity()(x), 1)
      qa_model = merge([dropout(question_output), dropout(answer_output)],
                       mode=similarity, output_shape=lambda _: (None, 1))
      # mode='cos', dot_axes=1)
      self._qa_model = Model(input=[self.question, self.get_answer()],
                             output=qa_model)
      print(self._qa_model.output_shape)

    return self._qa_model

  def get_train_model(self):
    if self._models is None:
      self._models = self.build()

    if self._qa_model is None:
      question_output, answer_output = self._models
      dropout = Dropout(self.similarity_params.get('similarity_dropout', 0.2))
      similarity = lambda x: K.expand_dims(
              K.relu(0.9 - self.get_similarity()(x)), 1)
      qa_model = merge([dropout(question_output), dropout(answer_output)],
                       mode=similarity, output_shape=lambda _: (None, 1))
      # mode='cos', dot_axes=1)
      self._qa_model = Model(input=[self.question, self.get_answer()],
                             output=qa_model)
      print(self._qa_model.output_shape)

    return self._qa_model

  def compile(self, optimizer, **kwargs):
    qa_model = self.get_qa_model()

    good_similarity = qa_model([self.question, self.answer_good])
    loss = self.get_train_model()([self.question, self.answer_good])

    # bad_similarity = qa_model([self.question, self.answer_bad])

    # loss = merge([good_similarity, bad_similarity],
    #              mode=lambda x: K.relu(self.config['margin'] - x[0] + x[1]),
    #              output_shape=lambda x: x[0])

    # loss = merge([good_similarity, good_similarity],
    #              mode=lambda x: K.relu(1 - x[0]),
    #              output_shape=lambda x: x[0])

    self.prediction_model = Model(input=[self.question, self.answer_good],
                                  output=good_similarity)
    self.prediction_model.compile(loss=lambda y_true, y_pred: y_pred,
                                  optimizer=optimizer, **kwargs)

    self.training_model = Model(input=[self.question, self.answer_good],
                                output=loss)
    self.training_model.compile(loss=lambda y_true, y_pred: y_pred,
                                optimizer=optimizer, **kwargs)

  def fit(self, x, y, **kwargs):
    assert self.training_model is not None, 'Must compile the model before fitting data'
    # y = np.zeros(shape=(x[0].shape[0],))  # doesn't get used
    return self.training_model.fit(x, y, **kwargs)

  def predict(self, x):
    assert self.prediction_model is not None and isinstance(self.prediction_model, Model)
    return self.prediction_model.predict_on_batch(x)

  def save_weights(self, file_name, **kwargs):
    assert self.prediction_model is not None, 'Must compile the model before saving weights'
    self.prediction_model.save_weights(file_name, **kwargs)

  def load_weights(self, file_name, **kwargs):
    assert self.prediction_model is not None, 'Must compile the model loading weights'
    self.prediction_model.load_weights(file_name, **kwargs)


class EmbeddingModel(LanguageModel):
  def build(self):
    question = self.question
    answer = self.get_answer()

    # add embedding layers
    weights = self.model_params.get('initial_embed_weights', None)
    weights = weights if weights is None else [weights]
    embedding = Embedding(input_dim=self.config['n_words'],
                          output_dim=self.model_params.get('n_embed_dims', 100),
                          # W_constraint=constraints.nonneg(),
                          weights=weights,
                          mask_zero=True)
    question_embedding = embedding(question)
    answer_embedding = embedding(answer)

    # maxpooling
    maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
    maxpool.__setattr__('supports_masking', True)
    question_pool = maxpool(question_embedding)
    answer_pool = maxpool(answer_embedding)

    return question_pool, answer_pool


class ConvolutionModel(LanguageModel):
  ### Validation loss at Epoch 65: 2.4e-6

  def build(self):
    assert self.config['question_len'] == self.config['answer_len']

    question = self.question
    answer = self.get_answer()

    # add embedding layers
    weights = self.model_params.get('initial_embed_weights', None)
    weights = weights if weights is None else [weights]
    embedding = Embedding(input_dim=self.config['n_words'],
                          output_dim=self.model_params.get('n_embed_dims', 100),
                          weights=weights)
    question_embedding = embedding(question)
    answer_embedding = embedding(answer)

    # turn off layer updating
    # embedding.params = []
    # embedding.updates = []

    # dense
    dense = TimeDistributed(Dense(self.model_params.get('n_hidden', 200),
                                  # activity_regularizer=regularizers.activity_l1(1e-4),
                                  # W_regularizer=regularizers.l1(1e-4),
                                  activation='tanh'))
    question_dense = dense(question_embedding)
    answer_dense = dense(answer_embedding)

    # cnn
    cnns = [Convolution1D(filter_length=filter_length,
                          nb_filter=self.model_params.get('nb_filters', 1000),
                          activation=self.model_params.get('conv_activation', 'relu'),
                          # W_regularizer=regularizers.l1(1e-4),
                          # activity_regularizer=regularizers.activity_l1(1e-4),
                          border_mode='same') for filter_length in [2, 3, 5, 7]]
    question_cnn = merge([cnn(question_dense) for cnn in cnns], mode='concat')
    answer_cnn = merge([cnn(answer_dense) for cnn in cnns], mode='concat')

    # maxpooling
    maxpool = Lambda(lambda x: K.max(x, axis=-1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
    avepool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=False), output_shape=lambda x: (x[0], x[2]))

    maxpool.__setattr__('supports_masking', True)
    avepool.__setattr__('supports_masking', True)
    question_pool = maxpool(question_cnn)
    answer_pool = maxpool(answer_cnn)

    return question_pool, answer_pool


class AttentionModel(LanguageModel):
  def build(self):
    question = self.question
    answer = self.get_answer()

    # add embedding layers
    weights = self.model_params.get('initial_embed_weights', None)
    weights = weights if weights is None else [weights]
    embedding = Embedding(input_dim=self.config['n_words'],
                          output_dim=self.model_params.get('n_embed_dims', 256),
                          # weights=weights,
                          mask_zero=True)
    question_embedding = embedding(question)
    answer_embedding = embedding(answer)

    # turn off layer updating
    # embedding.params = []
    # embedding.updates = []

    # question rnn part
    f_rnn = LSTM(self.model_params.get('n_lstm_dims', 141), return_sequences=True, dropout_U=0.2,
                 consume_less='mem')
    b_rnn = LSTM(self.model_params.get('n_lstm_dims', 141), return_sequences=True, dropout_U=0.2,
                 consume_less='mem', go_backwards=True)
    question_f_rnn = f_rnn(question_embedding)
    question_b_rnn = b_rnn(question_embedding)

    # maxpooling
    maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
    avepool = Lambda(lambda x: K.mean(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))

    # otherwise, it will raise a exception like:
    # Layer lambda_1 does not
    # support masking, but was passed an input_mask: Elemwise{neq,no_inplace}.0
    maxpool.__setattr__('supports_masking', True)
    avepool.__setattr__('supports_masking', True)

    question_pool = merge([maxpool(question_f_rnn), maxpool(question_b_rnn)], mode='concat', concat_axis=-1)

    # answer rnn part
    f_rnn = AttentionLSTM(self.model_params.get('n_lstm_dims', 141), question_pool, return_sequences=True,
                          consume_less='mem', single_attention_param=True)
    b_rnn = AttentionLSTM(self.model_params.get('n_lstm_dims', 141), question_pool, return_sequences=True,
                          consume_less='mem', go_backwards=True, single_attention_param=True)
    answer_f_rnn = f_rnn(answer_embedding)
    answer_b_rnn = b_rnn(answer_embedding)
    answer_pool = merge([maxpool(answer_f_rnn), maxpool(answer_b_rnn)], mode='concat', concat_axis=-1)

    return question_pool, answer_pool
