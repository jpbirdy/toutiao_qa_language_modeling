from __future__ import print_function

import os
import pandas as pd
import numpy as np
import sys
import random
from time import strftime, gmtime, localtime, sleep

import pickle

from gensim import corpora, models, similarities
from gensim.models import Word2Vec

random.seed(42)


class Evaluator:
  def __init__(self, conf=None):
    data_path = './toutiao_qa_python/pkl'
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


    self.valid_set = None
    self.train_set = None

    self.split_valid_set()

    self.tfidf = None
    self.question_words_tfidf_feas = None
    self.user_words_tfidf_feas = None
    self.words_tfidf()

    self.character_tfidf = None
    self.question_characters_tfidf_feas = None
    self.user_characters_tfidf_feas = None

    self.characters_tfidf()


  ##### Resources #####
  def load_w2v(self, name):
    return Word2Vec.load('%s_%d.m' % (name, self.w2v_len))

  def load(self, name):
    return pickle.load(open(os.path.join(self.path, name), 'rb'))

  ##### Building train&valid data set #####
  def split_valid_set(self):
    split = self.params.get('validation_split', 0)
    train_group = self.training_set.groupby('question_id')

    questions = [x[0] for x in list(train_group)]
    len_split = int(len(questions) * split)
    valid_questions = random.sample(questions, len_split)
    is_valid = lambda x: valid_questions.__contains__(x)
    self.training_set['is_valid'] = self.training_set['question_id'].apply(
            is_valid)

    self.valid_set = self.training_set[self.training_set.is_valid == True]
    self.train_set = self.training_set[self.training_set.is_valid == False]

  ##### Words tfidf #####
  def words_tfidf(self):
    question_words = [np.array(x, dtype=np.str) for x in
                               self.question_info['words_seq']]
    user_words = [np.array(x, dtype=np.str) for x in
                  self.user_info['user_desc_words_sec']]
    words = np.concatenate((question_words, user_words), axis=0)
    dictionary = corpora.Dictionary(words)
    corpus = [dictionary.doc2bow(text) for text in words]
    self.tfidf = models.TfidfModel(corpus)
    question_words_tfidf = self.tfidf[[dictionary.doc2bow(text) for text in
                             question_words]]
    user_words_tfidf = self.tfidf[[dictionary.doc2bow(text) for text in
                             user_words]]


    get_mean = lambda x : np.array(x)[::,1].mean() if len(x) > 0 else 0.0
    get_max = lambda x : np.array(x)[::,1].max() if len(x) > 0 else 0.0
    get_min = lambda x : np.array(x)[::,1].min() if len(x) > 0 else 0.0

    # (0:mean, 1:max, 2:min)
    self.question_words_tfidf_feas = [(get_mean(x), get_max(x), get_min(x))
                                      for x in
                                      question_words_tfidf]
    self.user_words_tfidf_feas = [(get_mean(x), get_max(x), get_min(x)) for x in
                                      user_words_tfidf]

  def characters_tfidf(self):
    question_characters = [np.array(x, dtype=np.str) for x in
                      self.question_info['character_seq']]

    user_characters = [np.array(x, dtype=np.str) for x in
                  self.user_info['user_desc_characters_sec']]
    words = np.concatenate((question_characters, user_characters), axis=0)
    dictionary = corpora.Dictionary(words)
    corpus = [dictionary.doc2bow(text) for text in words]
    self.character_tfidf = models.TfidfModel(corpus)
    question_characters_tfidf = self.character_tfidf[[dictionary.doc2bow(text)
                                                      for text in
                                       question_characters]]
    user_characters_tfidf = self.character_tfidf[[dictionary.doc2bow(text)
                                                  for text in
                                   user_characters]]

    get_mean = lambda x: np.array(x)[::, 1].mean() if len(x) > 0 else 0.0
    get_max = lambda x: np.array(x)[::, 1].max() if len(x) > 0 else 0.0
    get_min = lambda x: np.array(x)[::, 1].min() if len(x) > 0 else 0.0

    # (0:mean, 1:max, 2:min)
    self.question_characters_tfidf_feas = [(get_mean(x), get_max(x), get_min(x))
                                      for x in
                                      question_characters_tfidf]
    self.user_characters_tfidf_feas = [(get_mean(x), get_max(x), get_min(x))
                                       for x in
                                  user_characters_tfidf]

  def deal_questions_feas(self):
    # question_features
    self.log('calcu question features')
    self.log('question_words_num')
    self.question_info['words_num'] = [len(x) for x in self.question_info[
      'words_seq']]
    self.log('characters_num')
    self.question_info['characters_num'] = [len(x) for x in self.question_info[
      'character_seq']]

    self.log('tfidf')
    self.question_info['words_tfidf_mean'] = \
      [x[0] for x in self.question_words_tfidf_feas]
    self.question_info['words_tfidf_max'] = \
      [x[1] for x in self.question_words_tfidf_feas]
    self.question_info['words_tfidf_min'] = \
      [x[2] for x in self.question_words_tfidf_feas]

    self.question_info['characters_tfidf_mean'] = \
      [x[0] for x in self.question_characters_tfidf_feas]
    self.question_info['characters_tfidf_max'] = \
      [x[1] for x in self.question_characters_tfidf_feas]
    self.question_info['characters_tfidf_min'] = \
      [x[2] for x in self.question_characters_tfidf_feas]

    self.log('question feas complete')


  def deal_user_feas(self):
    self.log('calcu user features')
    self.user_info['tag_num'] = [len(str(x).split('/')) for x in
                                 self.user_info['user_tags']]
    self.log('user_words_num')
    self.user_info['words_num'] = [len(x) for x in self.user_info[
      'user_desc_words_sec']]
    self.log('characters_num')
    self.user_info['characters_num'] = [len(x) for x in self.user_info[
      'user_desc_characters_sec']]

    train_set_group = self.train_set.groupby('user_id')
    user_reply_history = {}

    self.log('calcu user reply history... It will take about 1 min')
    for u in train_set_group:
      user_id = u[0]
      reply_num = len(u[1][u[1].answer_flag == 1])
      ignore_num = len(u[1][u[1].answer_flag == 0])
      total_num = len(u[1])
      user_reply_history[user_id] = (reply_num, ignore_num, total_num,
                                     reply_num*1.0/total_num,
                                     ignore_num*1.0/total_num)

    self.log('calcu finished, calcu user history feas')
    get_user_reply = lambda x: user_reply_history[x][0] if \
      user_reply_history.has_key(x) else 0
    get_user_ignore = lambda x: user_reply_history[x][1] if \
      user_reply_history.has_key(x) else 0
    get_user_total = lambda x: user_reply_history[x][2] if \
      user_reply_history.has_key(x) else 0
    get_user_reply_rate = lambda x: user_reply_history[x][3] if \
      user_reply_history.has_key(x) else 0.0
    get_user_ignore_rate = lambda x: user_reply_history[x][4] if \
      user_reply_history.has_key(x) else 0.0

    self.user_info['user_id'] = self.user_info.index

    self.user_info['reply_num'] = self.user_info['user_id'].apply(get_user_reply)
    self.user_info['ignore_num'] = self.user_info['user_id'].apply(
            get_user_ignore)
    self.user_info['total_num'] = self.user_info['user_id'].apply(
            get_user_total)

    self.user_info['reply_rate'] = self.user_info['user_id'].apply(
            get_user_reply_rate)
    self.user_info['ignore_rate'] = self.user_info['user_id'].apply(
            get_user_ignore_rate)

    self.log('tfidf feas')

    self.user_info['words_tfidf_mean'] = \
      [x[0] for x in self.user_words_tfidf_feas]
    self.user_info['words_tfidf_max'] = \
      [x[1] for x in self.user_words_tfidf_feas]
    self.user_info['words_tfidf_min'] = \
      [x[2] for x in self.user_words_tfidf_feas]

    self.user_info['characters_tfidf_mean'] = \
      [x[0] for x in self.user_characters_tfidf_feas]
    self.user_info['characters_tfidf_max'] = \
      [x[1] for x in self.user_characters_tfidf_feas]
    self.user_info['characters_tfidf_min'] = \
      [x[2] for x in self.user_characters_tfidf_feas]

    self.log('user feas complete')

  def deal_question_user_feas(self):

    self.log('start q2u features')

    self.log('tag_w2v_sim fea')
    self.train_set['tag_w2v_sim'] = \
      [self.tag_w2v.n_similarity([str(self.question_info['question_tag'][test[
        0]])], self.user_info['user_tags'][test[1]].split('/'))
       for test in self.train_set.values]

    self.log('words_w2v_sim fea')
    self.train_set['words_w2v_sim'] = \
      [self.word_w2v.n_similarity(
              np.array(self.question_info['words_seq'][test[0]], dtype=str),
              np.array(self.user_info['user_desc_words_sec'][test[1]],
                       dtype=str))
       for test in self.train_set.values]

    self.log('characters_w2v_sim fea')
    self.train_set['characters_w2v_sim'] = \
      [self.character_w2v.n_similarity(
              np.array(self.question_info['character_seq'][test[0]], dtype=str),
              np.array(self.user_info['user_desc_characters_sec'][test[1]],
                       dtype=str))
       for test in self.train_set.values]

  def building_features(self):
    pass


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


  def log(self, str):
    self.print_time()
    print(str)


  def train(self, model):
    save_every = self.params.get('save_every', None)
    batch_size = self.params.get('batch_size', 128)
    nb_epoch = self.params.get('nb_epoch', 10)

    # tag w2v sim
    self.training_set['tag_w2v_sim'] =\
          [self.tag_w2v.n_similarity([str(self.question_info['question_tag'][test[
            0]])], self.user_info['user_tags'][test[1]].split('/'))
           for test in self.training_set.values]
    #


    val_ndcg = {'ndcg':0, 'epoch':0}

    self.save_conf()

    for i in range(1, nb_epoch):
      print('Epoch %d :: ' % i, end='')
      self.print_time()


      if save_every is not None and i % save_every == 0:
        self.save_epoch(model, i)

    # return val_loss
    return val_ndcg





