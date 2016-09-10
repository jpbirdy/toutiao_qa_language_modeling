from __future__ import print_function

import os
import pandas as pd
import sys
import random
from time import strftime, gmtime, localtime, sleep

import pickle

from gensim.models import Word2Vec


class Evaluator:
  def __init__(self, conf=None):
    data_path = './toutiao_qa_python/pkl'
    # sys.exit(1)
    self.path = data_path
    self.conf = dict() if conf is None else conf
    self.params = conf.get('training_params', dict())

    self.training_set = self.load('invited_info_train.pkl')
    self.question_info = self.load('question_info.pkl')
    self.user_info = self.load('user_info.pkl')
    self.w2v_len = conf.get('w2v_len', 100)
    self.tag_w2v = self.load_w2v('w2v_tag_embending')
    self.word_w2v = self.load_w2v('w2v_word_embending')
    self.character_w2v = self.load_w2v('w2v_character_embending')

  ##### Resources #####

  def load_w2v(self, name):
    return Word2Vec.load('%s_%d.m' % (name, self.w2v_len))

  def load(self, name):
    return pickle.load(open(os.path.join(self.path, name), 'rb'))

  ##### Loading / saving #####

  def save_conf(self):
    if not os.path.exists('models/'):
      os.makedirs('models/')
    if not os.path.exists('models/%s/' % self.conf.get('model_dir')):
      os.makedirs('models/%s/' % self.conf.get('model_dir'))
    sleep(1)
    pickle.dump(self.conf, open('models/%s/conf' % self.conf.get(
            'model_dir'), 'wb'))

  def save_epoch(self, model, epoch):
    if not os.path.exists('models/'):
      os.makedirs('models/')
    if not os.path.exists('models/%s/' % self.conf.get('model_dir')):
      os.makedirs('models/%s/' % self.conf.get('model_dir'))
    model.save_weights('models/%s/weights_epoch_%d.h5' %
                       (self.conf.get('model_dir'), epoch), overwrite=True)

  def load_epoch(self, model, epoch):
    assert os.path.exists('models/%s/weights_epoch_%d.h5' %
                          (self.conf.get('model_dir'), epoch)), \
      'Weights at epoch %d not found' % epoch
    model.load_weights('models/%s/weights_epoch_%d.h5' %
                          (self.conf.get('model_dir'), epoch))

  ##### Training #####

  @staticmethod
  def print_time():
    print(strftime('%Y-%m-%d %H:%M:%S :: ', localtime()), end='')

  def train(self, model):
    save_every = self.params.get('save_every', None)
    batch_size = self.params.get('batch_size', 128)
    nb_epoch = self.params.get('nb_epoch', 10)
    split = self.params.get('validation_split', 0)


    self.training_set['tag_w2v_sim'] =\
          [self.tag_w2v.n_similarity([str(self.question_info['question_tag'][test[
            0]])], self.user_info['user_tags'][test[1]].split('/'))
           for test in self.training_set.values]

    # questions = list()
    # answers = list()

    train_group = self.training_set.groupby('question_id')

    all_users = list(self.user_info.index)

    question_ids = list()
    good_answer_ids = list()
    bad_answer_ids = list()

    questions = [x[0] for x in list(train_group)]
    len_split = int(len(questions) * split)
    valid_questions = random.sample(questions, len_split)

    valid_qid = list()
    valid_uid = list()
    valid_answer = list()

    for x in list(train_group):
      question_id = x[0]
      answer_info = x[1]
      if valid_questions.__contains__(question_id):
        for info in answer_info.values:
          valid_qid.append(info[0])
          valid_uid.append(info[1])
          valid_answer.append(info[2])
      else:
        good_bad = [(g, b) for g in answer_info['user_id'][
            answer_info.answer_flag == 1] for b in answer_info['user_id'][
            answer_info.answer_flag == 0]]
        for gb in good_bad:
          question_ids.append(question_id)
          good_answer_ids.append(gb[0])
          bad_answer_ids.append(gb[1])
          bad_sample = random.sample(all_users, bad_answer_sample)
          for bad in bad_sample:
            question_ids.append(question_id)
            good_answer_ids.append(gb[0])
            bad_answer_ids.append(bad)

    sample = self.conf.get('sample')
    if sample > 0:
      print('Selected sample, num is %d' % sample)
      sample = random.sample(range(len(question_ids)), sample)
      question_ids = [question_ids[s] for s in sample]
      good_answer_ids = [good_answer_ids[s] for s in sample]
      bad_answer_ids = [bad_answer_ids[s] for s in sample]

    # val_loss = {'loss': 1., 'epoch': 0}
    val_ndcg = {'ndcg':0, 'epoch':0}

    self.save_conf()

    for i in range(1, nb_epoch):
      # sample from all answers to get bad answers
      print('Epoch %d :: ' % i, end='')
      self.print_time()

      # valid_set['predict'] = [x[0][0] for x in predict]
      # valid_group = valid_set.groupby('qid')

      # scores = list()

      # for x in list(valid_group):
      #   answer_info = x[1].sort('predict', ascending=False)
      #   predict = [answer_info['predict'][x] * answer_info['answer_flag'][x]
      #              for x in answer_info.index]
      #   from ndcg import ndcg_at_k
      #   scores.append(
      #     ndcg_at_k(predict, 5) * 0.5 + ndcg_at_k(predict, 10) * 0.5)
      #
      # valid_ndcg = np.mean(scores)
      # print('ndcg mean is %lf' % valid_ndcg)
      # if valid_ndcg > val_ndcg['ndcg']:
      #   val_ndcg = {'ndcg': valid_ndcg, 'epoch':i}
      #
      # print('Best: Ndcg = {}, Epoch = {}'.format(val_ndcg['ndcg'],
      #                                            val_ndcg['epoch']))

      # if hist.history['val_loss'][0] < val_loss['loss']:
      #   val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
      # print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'],
      #                                            val_loss['epoch']))

      if save_every is not None and i % save_every == 0:
        self.save_epoch(model, i)

    # return val_loss
    return val_ndcg




good_sim = [w2v.n_similarity([str(question_info['question_tag'][test[0]])],
                             user_info['user_tags'][test[1]].split('/')) for
            test
            in training_set.values]