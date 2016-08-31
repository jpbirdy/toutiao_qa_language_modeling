# coding=utf-8
from __future__ import print_function

import os

import sys
import random
from time import strftime, gmtime

import pickle

import pandas
from keras.optimizers import Adam
from scipy.stats import rankdata

from toutiao_qa_eval import Evaluator
from keras_models import EmbeddingModel, AttentionModel, ConvolutionModel

random.seed(42)


class EvaluatorEval(Evaluator):

  def now_ds(self):
    import time
    # 获得当前时间时间戳
    now = int(time.time())
    # 转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
    timeArray = time.localtime(now)
    otherStyleTime = time.strftime("%Y-%m-%d %H_%M_%S", timeArray)
    return otherStyleTime

  def predict(self, model):
    batch_size = self.params.get('batch_size', 128)

    import pandas as pd
    valid_set = pd.read_csv('toutiao_qa_python/validate_nolabel.txt')

    question_info = self.load('question_info.pkl')
    user_info = self.load('user_info.pkl')

    question_words_seq = [
      list(question_info['words_seq'][x])
      for x in valid_set['qid']]

    # questions = list()
    # answers = list()

    answers_words_seq = [
      list(user_info['user_desc_words_sec'][x])
      for x in valid_set['uid']]

    question_words_seq = self.padq(question_words_seq)
    answers_words_seq = self.pada(answers_words_seq)

    predict = model.prediction_model.predict(
                    [question_words_seq, answers_words_seq],
                    batch_size=batch_size, verbose=1)
    output = []
    for i in valid_set.index:
      output.append([valid_set['qid'][i], valid_set['uid'][i], predict[i]])

    import csv
    output_file = open('output/valid_%s.csv' % self.now_ds(), 'w')
    writer = csv.writer(output_file)
    writer.writerow(['qid', 'uid', 'label'])
    for x in output:
      writer.writerow([x[0], x[1], x[2][0][0]])

    output_file.close()

    return output

  def valid(self, model):
    batch_size = self.params.get('batch_size', 128)

    invited_info_train = self.load('invited_info_train.pkl')

    question_info = self.load('question_info.pkl')
    user_info = self.load('user_info.pkl')

    question_words_seq = [
      list(question_info['words_seq'][x])
      for x in invited_info_train['question_id']]

    # questions = list()
    # answers = list()

    answers_words_seq = [
      list(user_info['user_desc_words_sec'][x])
      for x in invited_info_train['user_id']]

    question_words_seq = self.padq(question_words_seq)
    answers_words_seq = self.pada(answers_words_seq)

    predict = model.prediction_model.predict(
            [question_words_seq, answers_words_seq],
            batch_size=batch_size, verbose=1)
    # output = []
    # for i in invited_info_train.index:
    #   output.append([invited_info_train['question_id'][i], invited_info_train[
    #     'user_id'][i], predict[i]])

    invited_info_train['predict'] = [x[0][0] for x in predict]
    train_group = invited_info_train.groupby('question_id')

    scores = list()

    for x in list(train_group):
      # question_id = x[0]
      answer_info = x[1].sort('predict', ascending=False)
      predict = [answer_info['predict'][x] * answer_info['answer_flag'][x]
                 for x in answer_info.index]
      from ndcg import ndcg_at_k
      scores.append(ndcg_at_k(predict, 5) * 0.5 + ndcg_at_k(predict, 10) * 0.5)

    print('ndcg mean is %lf' % np.mean(scores))

    return train_group

if __name__ == '__main__':
  import numpy as np

  model_dir = '.'                      #0.02aesd,epoch4 local total 0.670610
  # model_dir = '2016-08-30 16:18:23'  #0.02gesd,epoch4 local total 0.646764
  # model_dir = '2016-08-30 16:43:19'  #0.1gesd,epoch4 local total 0.612545
  # model_dir = '2016-08-30 17:08:44'  #0.5gesd,epoch4 local total 0.606227
  # model_dir = '2016-08-30 18:01:43'    #0.5aesd,epoch4 local total 0.590244
  # sim_type = 'gesd'
  sim_type = 'aesd'
  epoch = 5


  conf = {
    'question_len': 50,
    'answer_len': 50,
    # 'n_words': 22353,  # len(vocabulary) + 1
    'n_words': 37813,  # len(vocabulary) + 1
    # 'margin': 0.02,
    'margin': 0.5,
    'sample': 0,
    'model_dir': model_dir,

    'training_params': {
      'save_every': 1,
      # 'batch_size': 20,
      'batch_size': 256,
      # 'batch_size': 1024,
      'nb_epoch': 50,
      # 'nb_epoch': 5,
      # 'validation_split': 0.,
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
      'mode': sim_type,
      # 'mode': 'gesd',
      # 'mode': 'aesd',
      'gamma': 1,
      'c': 1,
      'd': 2,
    }
  }

  evaluator = EvaluatorEval(conf)

  ##### Define model ######
  model = AttentionModel(conf)
  optimizer = conf.get('training_params', dict()).get('optimizer', 'rmsprop')
  model.compile(optimizer=optimizer)

  # save embedding layer
  # evaluator.load_epoch(model, 7)
  # embedding_layer = model.prediction_model.layers[2].layers[2]
  # weights = embedding_layer.get_weights()[0]
  # np.save(open('models/embedding_1000_dim.h5', 'wb'), weights)

  # train the model
  evaluator.load_epoch(model, epoch)
  output = evaluator.valid(model)
  # output = evaluator.predict(model)

  # print(output)





