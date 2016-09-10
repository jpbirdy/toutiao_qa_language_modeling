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
    output_file = open('output/valid.csv', 'w')
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
      question_id = x[0]
      answer_info = x[1].sort('predict', ascending=False)
      predict = [answer_info['predict'][x] * answer_info['answer_flag'][x]
                 for x in answer_info.index]
      from ndcg import ndcg_at_k
      scores.append(ndcg_at_k(predict, 5) * 0.5 + ndcg_at_k(predict, 10) * 0.5)

    print('ndcg mean is %lf' % np.mean(scores))

    return train_group

if __name__ == '__main__':
  import numpy as np

  # model_dir = '.'  # 0.02aesd,epoch4 local total 0.677434
                   #
  # model_dir = '2016-08-30 16:18:23'  #0.02gesd,epoch4 local total 0.646764
                                       #epoch5 local total 0.649898
  # model_dir = '2016-08-30 16:43:19'  #0.1gesd,epoch4 local total 0.612545
  # model_dir = '2016-08-30 17:08:44'  #0.5gesd,epoch4 local total 0.606227
  # model_dir = '2016-08-30 18:01:43'  #0.5aesd,epoch4 local total 0.590244
  # model_dir = '2016-08-31 17:15:11'  #modify param 0.02aesd,epoch1 0.637881
                                     #2  0.668108
                                     #3  0.678860
                                     #4  0.68599
                                     #9  0.705385
  model_dir = '2016-09-09 09:19:06'
  epoch = 1

  conf = pickle.load(open('models/%s/conf' % model_dir, 'rb'))

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





