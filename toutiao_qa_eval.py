from __future__ import print_function

import os

import sys
import random
from time import strftime, gmtime

import pickle

from keras.optimizers import Adam
from scipy.stats import rankdata

from keras_models import EmbeddingModel, AttentionModel, ConvolutionModel

random.seed(42)


class Evaluator:
  def __init__(self, conf=None):
    data_path = './toutiao_qa_python/pkl'
    # sys.exit(1)
    self.path = data_path
    self.conf = dict() if conf is None else conf
    self.params = conf.get('training_params', dict())
    # self.answers = self.load('answers')  # self.load('generated')
    self._eval_sets = None

  ##### Resources #####

  def load(self, name):
    return pickle.load(open(os.path.join(self.path, name), 'rb'))

  ##### Loading / saving #####

  def save_epoch(self, model, epoch):
    if not os.path.exists('models/'):
      os.makedirs('models/')
    model.save_weights('models/weights_epoch_%d.h5' % epoch, overwrite=True)

  def load_epoch(self, model, epoch):
    assert os.path.exists('models/weights_epoch_%d.h5' % epoch), \
      'Weights at epoch %d not found' % epoch
    model.load_weights('models/weights_epoch_%d.h5' % epoch)

  ##### Padding #####

  def padq(self, data):
    return self.pad(data, self.conf.get('question_len', None))

  def pada(self, data):
    return self.pad(data, self.conf.get('answer_len', None))

  def pad(self, data, len=None):
    from keras.preprocessing.sequence import pad_sequences
    return pad_sequences(data, maxlen=len, padding='post',
                         truncating='post', value=0)

  ##### Training #####

  def print_time(self):
    print(strftime('%Y-%m-%d %H:%M:%S :: ', gmtime()), end='')

  def train(self, model):
    save_every = self.params.get('save_every', None)
    batch_size = self.params.get('batch_size', 128)
    nb_epoch = self.params.get('nb_epoch', 10)
    split = self.params.get('validation_split', 0)

    training_set = self.load('invited_info_train.pkl')
    question_info = self.load('question_info.pkl')
    user_info = self.load('user_info.pkl')

    question_words_seq = [
        list(question_info['words_seq'][x])
        for x in training_set['question_id']]

    # questions = list()
    # answers = list()

    answers_words_seq = [
        list(user_info['user_desc_words_sec'][x])
        for x in training_set['user_id']]

    y = np.array(list(training_set['answer_flag']))

    # questions = self.padq(questions)
    question_words_seq = self.padq(question_words_seq)
    answers_words_seq = self.pada(answers_words_seq)

    val_loss = {'loss': 1., 'epoch': 0}

    for i in range(1, nb_epoch):
      # sample from all answers to get bad answers
      print('Epoch %d :: ' % i, end='')
      self.print_time()
      hist = model.fit([question_words_seq, answers_words_seq], y, nb_epoch=1,
                       batch_size=batch_size,
                       validation_split=split)

      if hist.history['val_loss'][0] < val_loss['loss']:
        val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
      print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'],
                                                 val_loss['epoch']))

      if save_every is not None and i % save_every == 0:
        self.save_epoch(model, i)

    return val_loss

  ##### Evaluation #####

  def prog_bar(self, so_far, total, n_bars=20):
    n_complete = int(so_far * n_bars / total)
    if n_complete >= n_bars - 1:
      print('\r[' + '=' * n_bars + ']', end='')
    else:
      s = '\r[' + '=' * (n_complete - 1) + '>' + \
          '.' * (n_bars - n_complete) + ']'
      print(s, end='')

  def eval_sets(self):
    if self._eval_sets is None:
      self._eval_sets = dict([(s, self.load(s)) for s in
                              ['dev', 'test1', 'test2']])
    return self._eval_sets

  def get_mrr(self, model, evaluate_all=False):
    top1s = list()
    mrrs = list()

    for name, data in self.eval_sets().items():
      if evaluate_all:
        self.print_time()
        print('----- %s -----' % name)

      random.shuffle(data)

      if not evaluate_all and 'n_eval' in self.params:
        data = data[:self.params['n_eval']]

      c_1, c_2 = 0, 0

      for i, d in enumerate(data):
        # if evaluate_all:
        #   self.prog_bar(i, len(data))
        self.prog_bar(i, len(data))
        indices = d['good'] + d['bad']
        answers = self.pada([i for i in indices])
        question = self.padq([d['question']] * len(indices))

        n_good = len(d['good'])
        sims = model.predict([question, answers])
        r = rankdata(sims, method='max')

        max_r = np.argmax(sims)
        max_n = np.argmax(sims[:n_good])

        # print(' '.join(self.revert(d['question'])))
        # print(' '.join(self.revert(self.answers[indices[max_r]])))
        # print(' '.join(self.revert(self.answers[indices[max_n]])))

        c_1 += 1 if max_r == max_n else 0
        c_2 += 1 / float(r[max_r] - r[max_n] + 1)

      top1 = c_1 / float(len(data))
      mrr = c_2 / float(len(data))

      del data

      if evaluate_all:
        print('Top-1 Precision: %f' % top1)
        print('MRR: %f' % mrr)

      top1s.append(top1)
      mrrs.append(mrr)

    # rerun the evaluation if above some threshold
    if not evaluate_all:
      print('Top-1 Precision: {}'.format(top1s))
      print('MRR: {}'.format(mrrs))
      evaluate_all_threshold = self.params.get('evaluate_all_threshold', dict())
      evaluate_mode = evaluate_all_threshold.get('mode', 'all')
      mrr_theshold = evaluate_all_threshold.get('mrr', 1)
      top1_threshold = evaluate_all_threshold.get('top1', 1)

      if evaluate_mode == 'any':
        evaluate_all = evaluate_all or any([x >= top1_threshold for x in top1s])
        evaluate_all = evaluate_all or any([x >= mrr_theshold for x in mrrs])
      else:
        evaluate_all = evaluate_all or all([x >= top1_threshold for x in top1s])
        evaluate_all = evaluate_all or all([x >= mrr_theshold for x in mrrs])

      if evaluate_all:
        return self.get_mrr(model, evaluate_all=True)

    return top1s, mrrs


if __name__ == '__main__':
  import numpy as np

  conf = {
    'question_len': 50,
    'answer_len': 50,
    # 'n_words': 22353,  # len(vocabulary) + 1
    'n_words': 37813,  # len(vocabulary) + 1
    'margin': 0.02,

    'training_params': {
      'save_every': 1,
      # 'batch_size': 20,
      'batch_size': 256,
      # 'nb_epoch': 50,
      'nb_epoch': 5,
      'validation_split': 0.1,
      'optimizer': Adam(
              clipnorm=1e-2),
    },

    'model_params': {
      # 'n_embed_dims': 100,
      'n_embed_dims': 256,
      'n_hidden': 200,

      # convolution
      'nb_filters': 1000,
      # * 4
      'conv_activation': 'tanh',

      # recurrent
      'n_lstm_dims': 141,
      # * 2

      # 'initial_embed_weights':
      #   np.load('./word2vec_100_dim.embeddings'),
      'similarity_dropout': 0.5,
    },

    'similarity_params': {
      # 'mode': 'gesd',
      'mode': 'aesd',
      'gamma': 1,
      'c': 1,
      'd': 2,
    }
  }

  evaluator = Evaluator(conf)

  ##### Define model ######
  model = AttentionModel(conf)
  optimizer = conf.get('training_params', dict()).get('optimizer', 'rmsprop')
  model.compile(optimizer=optimizer, metrics=['accuracy'])

  # save embedding layer
  # evaluator.load_epoch(model, 7)
  # embedding_layer = model.prediction_model.layers[2].layers[2]
  # weights = embedding_layer.get_weights()[0]
  # np.save(open('models/embedding_1000_dim.h5', 'wb'), weights)

  # train the model
  # evaluator.load_epoch(model, 1)
  # print('load model success.')
  best_loss = evaluator.train(model)



  # evaluate mrr for a particular epoch
  # evaluator.load_epoch(model, best_loss['epoch'])
  # evaluator.load_epoch(model, 31)
  # evaluator.get_mrr(model, evaluate_all=True)
  # mrr = evaluator.get_mrr(model, evaluate_all=False)
  # print(mrr)
  # for epoch in range(1, 100):
  #     print('Epoch %d' % epoch)
  #     evaluator.load_epoch(model, epoch)
  #     evaluator.get_mrr(model, evaluate_all=True)
