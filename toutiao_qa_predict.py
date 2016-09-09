from __future__ import print_function

import os

import sys
import random
from time import strftime, gmtime

import pickle

import pandas
from keras.optimizers import Adam
from scipy.stats import rankdata

from keras_models_w2v import AttentionModelW2V
from toutiao_qa_eval import Evaluator

random.seed(42)


class EvaluatorEval(Evaluator):

  def predict(self, model):
    batch_size = 128

    import pandas as pd
    valid_set = pd.read_csv('toutiao_qa_python/validate_nolabel.txt')

    question_info = self.load('question_info.pkl')
    user_info = self.load('user_info.pkl')

    print('start sequences padding')

    question_info['words_seq_padding'] = list(self.padq(list(question_info[
                                                               'words_seq'])))
    user_info['user_desc_words_sec_padding'] = list(self.pada(list(user_info[
                                                                     'user_desc_words_sec'])))
    print('start word2vec mapping')

    get_w2v = lambda w: [
      list(self.w2v[str(x)] if self.w2v.__contains__(str(x)) else np.zeros(
              self.w2v_len))
      for x in w]

    question_info['words_seq_padding_w2v'] = \
      question_info['words_seq_padding'].apply(get_w2v)
    user_info['user_desc_words_sec_padding_w2v'] = \
      user_info['user_desc_words_sec_padding'].apply(get_w2v)


    question_words_seq = np.array([
      list(question_info['words_seq_padding_w2v'][x])
      for x in valid_set['qid']])

    # questions = list()
    # answers = list()

    answers_words_seq = np.array([
      list(user_info['user_desc_words_sec_padding_w2v'][x])
      for x in valid_set['uid']])

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
    batch_size = 128

    invited_info_train = self.load('invited_info_train.pkl')

    question_info = self.load('question_info.pkl')
    user_info = self.load('user_info.pkl')

    print('start sequences padding')

    question_info['words_seq_padding'] = list(self.padq(list(question_info[
                                                               'words_seq'])))
    user_info['user_desc_words_sec_padding'] = list(self.pada(list(user_info[
                                                                     'user_desc_words_sec'])))
    print('start word2vec mapping')

    get_w2v = lambda w: [
      list(self.w2v[str(x)] if self.w2v.__contains__(str(x)) else np.zeros(
              self.w2v_len))
      for x in w]

    question_info['words_seq_padding_w2v'] = \
      question_info['words_seq_padding'].apply(get_w2v)
    user_info['user_desc_words_sec_padding_w2v'] = \
      user_info['user_desc_words_sec_padding'].apply(get_w2v)

    question_words_seq = np.array([
      list(question_info['words_seq_padding_w2v'][x])
      for x in invited_info_train['question_id']])

    # questions = list()
    # answers = list()

    answers_words_seq = np.array([
      list(user_info['user_desc_words_sec_padding_w2v'][x])
      for x in invited_info_train['user_id']])

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

  model_dir = '2016-09-09 11:36:33'
  epoch = 5
  conf = pickle.load(open('models/%s/conf' % model_dir, 'rb'))

  evaluator = EvaluatorEval(conf)
  model = AttentionModelW2V(conf)
  # model = evaluator.load_model(model_dir)
  evaluator.load_epoch(model, epoch)
  output = evaluator.valid(model)
  # output = evaluator.predict(model)

  # print(output)





