# coding=utf-8
from __future__ import print_function

import os
import pandas as pd
import numpy as np
import sys
import random
from time import strftime, gmtime, localtime, sleep

import pickle


import xgboost as xgb
from xgboost import DMatrix
from ndcg import ndcg_at_k
random.seed(42)


class XGMatrix(DMatrix):
  def __init__(self, data, label=None, missing=None,
                 weight=None, silent=False,
                 feature_names=None, feature_types=None):
    DMatrix.__init__(self, data, label, missing, weight, silent,
                 feature_names, feature_types)
    self.group = None

  def set_group(self, group):
    super(XGMatrix, self).set_group(group)
    self.group = group

  def get_group(self):
    return self.group


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
    self.valid_set_group = None
    self.train_set = None
    self.train_set_group = None

    self.split_valid_set()

    self.tfidf = None
    self.question_words_tfidf_feas = None
    self.user_words_tfidf_feas = None


    self.character_tfidf = None
    self.question_characters_tfidf_feas = None
    self.user_characters_tfidf_feas = None

    self.bst = None


    #   submit result
    import pandas as pd
    self.submit_valid_set = pd.read_csv(
            'toutiao_qa_python/validate_nolabel.txt')
    if not self.load_features():
      self.building_features()
    else:
      self.log('loadding features from pickle')


  ##### Resources #####
  def load_w2v(self, name):
    from gensim.models import Word2Vec
    return Word2Vec.load('%s_%d.m' % (name, self.w2v_len))

  def load(self, name):
    return pickle.load(open(os.path.join(self.path, name), 'rb'))

  def load_feature(self, name):
    return pickle.load(open(name, 'rb'))

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
    from gensim import corpora, models, similarities
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
    from gensim import corpora, models, similarities
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
    # remove nan
    self.train_set['tag_w2v_sim'] = [x if type(x) == np.float64 else 0.0 for x
                                       in \
                                       self.train_set['tag_w2v_sim']]


    self.valid_set['tag_w2v_sim'] = \
      [self.tag_w2v.n_similarity([str(self.question_info['question_tag'][test[
        0]])], self.user_info['user_tags'][test[1]].split('/'))
       for test in self.valid_set.values]
    # remove nan
    self.valid_set['tag_w2v_sim'] = [x if type(x) == np.float64 else 0.0 for x
                                     in \
                                     self.valid_set['tag_w2v_sim']]

    self.log('words_w2v_sim fea')
    self.train_set['words_w2v_sim'] = \
      [self.word_w2v.n_similarity(
              np.array(self.question_info['words_seq'][test[0]], dtype=str),
              np.array(self.user_info['user_desc_words_sec'][test[1]],
                       dtype=str))
       for test in self.train_set.values]
    self.train_set['words_w2v_sim'] = [x if type(x) == np.float64 else 0.0 for x
                                     in \
                                     self.train_set['words_w2v_sim']]


    self.valid_set['words_w2v_sim'] = \
      [self.word_w2v.n_similarity(
              np.array(self.question_info['words_seq'][test[0]], dtype=str),
              np.array(self.user_info['user_desc_words_sec'][test[1]],
                       dtype=str))
       for test in self.valid_set.values]
    self.valid_set['words_w2v_sim'] = [x if type(x) == np.float64 else 0.0 for x
                                       in \
                                       self.valid_set['words_w2v_sim']]


    self.log('characters_w2v_sim fea')
    self.train_set['characters_w2v_sim'] = \
      [self.character_w2v.n_similarity(
              np.array(self.question_info['character_seq'][test[0]], dtype=str),
              np.array(self.user_info['user_desc_characters_sec'][test[1]],
                       dtype=str))
       for test in self.train_set.values]
    self.train_set['characters_w2v_sim'] = [x if type(x) == np.float64 else 0.0 for x
                                       in \
                                       self.train_set['characters_w2v_sim']]

    self.valid_set['characters_w2v_sim'] = \
      [self.character_w2v.n_similarity(
              np.array(self.question_info['character_seq'][test[0]], dtype=str),
              np.array(self.user_info['user_desc_characters_sec'][test[1]],
                       dtype=str))
       for test in self.valid_set.values]
    self.valid_set['characters_w2v_sim'] = [x if type(x) == np.float64 else 0.0
                                            for x
                                            in \
                                            self.valid_set[
                                              'characters_w2v_sim']]

    user_features = {
      'user_tag_num' : lambda x : self.user_info['tag_num'][x],
      'user_words_num' : lambda x : self.user_info['words_num'][x],
      'user_characters_num' : lambda x : self.user_info['characters_num'][x],
      'user_reply_num' : lambda x: self.user_info['reply_num'][x],
      'user_ignore_num' : lambda x : self.user_info['ignore_num'][x],
      'user_total_num' : lambda x : self.user_info['total_num'][x],
      'user_reply_rate' : lambda x : self.user_info['reply_rate'][x],
      'user_ignore_rate' : lambda x : self.user_info['ignore_rate'][x],
      'user_characters_tfidf_mean' : lambda x : self.user_info[
        'characters_tfidf_mean'][x],
      'user_characters_tfidf_max' : lambda x : self.user_info[
        'characters_tfidf_max'][x],
      'user_characters_tfidf_min' : lambda x : self.user_info[
        'characters_tfidf_min'][x],
    }

    for key, func in user_features.items():
      self.train_set[key] = self.train_set['user_id'].apply(func)

    for key, func in user_features.items():
      self.valid_set[key] = self.valid_set['user_id'].apply(func)

    question_features = {
      'question_point_num': lambda x: self.question_info['point_num'][x],
      'question_reply_num': lambda x: self.question_info['reply_num'][x],
      'question_good_reply_num': lambda x: self.question_info['good_reply_num'][x],
      'question_words_num': lambda x: self.question_info['words_num'][x],
      'question_characters_num': lambda x: self.question_info['characters_num'][x],
      'question_words_tfidf_mean': lambda x: self.question_info['words_tfidf_mean'][x],
      'question_words_tfidf_max': lambda x: self.question_info['words_tfidf_max'][x],
      'question_words_tfidf_min': lambda x: self.question_info['words_tfidf_min'][x],
      'question_characters_tfidf_mean': lambda x: self.question_info[
        'characters_tfidf_mean'][x],
      'question_characters_tfidf_max': lambda x: self.question_info[
        'characters_tfidf_max'][x],
      'question_characters_tfidf_min': lambda x: self.question_info[
        'characters_tfidf_min'][x],
    }

    for key, func in question_features.items():
      self.train_set[key] = self.train_set['question_id'].apply(func)

    for key, func in question_features.items():
      self.valid_set[key] = self.valid_set['question_id'].apply(func)

    self.train_set = self.train_set.sort_values('question_id')
    self.valid_set = self.valid_set.sort_values('question_id')
    self.train_set_group = self.train_set.groupby('question_id').size()
    self.valid_set_group = self.valid_set.groupby('question_id').size()


    # online submit set
    self.submit_valid_set['tag_w2v_sim'] = \
      [self.tag_w2v.n_similarity([str(self.question_info['question_tag'][test[
        0]])], self.user_info['user_tags'][test[1]].split('/'))
       for test in self.submit_valid_set.values]
    # remove nan
    self.submit_valid_set['tag_w2v_sim'] = [x if type(x) == np.float64 else 0.0
                                            for x
                                            in \
                                            self.submit_valid_set[
                                              'tag_w2v_sim']]

    self.submit_valid_set['words_w2v_sim'] = \
      [self.word_w2v.n_similarity(
              np.array(self.question_info['words_seq'][test[0]], dtype=str),
              np.array(self.user_info['user_desc_words_sec'][test[1]],
                       dtype=str))
       for test in self.submit_valid_set.values]
    self.submit_valid_set['words_w2v_sim'] = [
      x if type(x) == np.float64 else 0.0 for x
      in \
      self.submit_valid_set['words_w2v_sim']]

    self.submit_valid_set['characters_w2v_sim'] = \
      [self.character_w2v.n_similarity(
              np.array(self.question_info['character_seq'][test[0]], dtype=str),
              np.array(self.user_info['user_desc_characters_sec'][test[1]],
                       dtype=str))
       for test in self.submit_valid_set.values]
    self.submit_valid_set['characters_w2v_sim'] = [
      x if type(x) == np.float64 else 0.0
      for x
      in \
      self.submit_valid_set[
        'characters_w2v_sim']]
    # for key, func in question_features.items():
    #   self.submit_valid_set[key] = self.submit_valid_set['qid'].apply(func)

    for key, func in user_features.items():
      self.submit_valid_set[key] = self.submit_valid_set['uid'].apply(func)

    for key, func in question_features.items():
      self.submit_valid_set[key] = self.submit_valid_set['qid'].apply(func)

  def building_features(self):
    self.log('start building features')
    self.words_tfidf()
    self.characters_tfidf()
    self.deal_questions_feas()
    self.deal_user_feas()
    self.deal_question_user_feas()



  ##### Loading / saving #####

  def save_features(self):
    if not os.path.exists('features/'):
      os.makedirs('features/')
    if not os.path.exists('features/%s/' % self.conf.get('version')):
      os.makedirs('features/%s/' % self.conf.get('version'))
    sleep(1)
    pickle.dump(self.train_set, open('features/%s/train_set.pkl' %
                                     self.conf.get('version'), 'wb'))
    pickle.dump(self.train_set_group, open('features/%s/train_set_group.pkl' %
                                     self.conf.get('version'), 'wb'))
    pickle.dump(self.valid_set, open('features/%s/valid_set.pkl' %
                                           self.conf.get('version'), 'wb'))
    pickle.dump(self.valid_set_group, open('features/%s/valid_set_group.pkl' %
                                     self.conf.get('version'), 'wb'))
    pickle.dump(self.submit_valid_set, open('features/%s/submit_valid_set.pkl' %
                                           self.conf.get('version'), 'wb'))

  def load_features(self):
    if os.path.exists('features/%s/' % self.conf.get('version')):
      self.train_set = self.load_feature('features/%s/train_set.pkl' %
                                         self.conf.get('version'))
      self.train_set_group = self.load_feature('features/%s/train_set_group.pkl' %
                                          self.conf.get('version'))
      self.valid_set = self.load_feature('features/%s/valid_set.pkl' %
                                               self.conf.get('version'))
      self.valid_set_group = self.load_feature('features/%s/valid_set_group.pkl' %
                                         self.conf.get('version'))
      self.submit_valid_set = self.load_feature('features/%s/submit_valid_set.pkl' %
                                         self.conf.get('version'))

      return True
    else :
      return False

  ##### Training #####

  @staticmethod
  def print_time():
    print(strftime('%Y-%m-%d %H:%M:%S :: ', localtime()), end='')


  def log(self, str):
    self.print_time()
    print(str)

  def ndcgerror(self, preds, dtrain):
    assert isinstance(dtrain, XGMatrix)
    labels = dtrain.get_label()
    group = dtrain.get_group()

    offset = 0
    scores = list()
    for size in group:
      p = preds[offset:offset+size]
      l = labels[offset:offset+size]
      offset += size

      rec_result = [(p[i] , l[i]) for i in range(size)]
      rec_result = sorted(rec_result, key=lambda x : x[0], reverse=True)
      predict = [x[1] for x in rec_result]
      scores.append(ndcg_at_k(predict, 5) * 0.5 + ndcg_at_k(predict, 10) * 0.5)
    return 'ndcg_error', np.mean(scores)

  def train(self):
    nb_epoch = self.params.get('nb_epoch', None)
    #
    label = 'answer_flag'
    features = list(self.train_set.keys())
    features.remove('question_id')
    features.remove('user_id')
    features.remove('answer_flag')
    features.remove('is_valid')

    train = self.train_set.drop(['question_id', 'user_id', 'answer_flag',
                                 'is_valid'], axis=1)
    valid = self.valid_set.drop(['question_id', 'user_id', 'answer_flag',
                                 'is_valid'], axis=1)


    dtrain = XGMatrix(data=train, label=self.train_set[label])
    dvalid = XGMatrix(data=valid, label=self.valid_set[label])



    # only use for ranking
    dtrain.set_group(list(self.train_set_group))
    dvalid.set_group(list(self.valid_set_group))

    watchlist = [(dtrain, 'train'),
                 (dvalid, 'valid')]

    self.bst = xgb.train(self.conf['xgb_param'], dtrain, nb_epoch, watchlist ,
                    early_stopping_rounds=2000,
                    feval=self.ndcgerror
                    )

    print('best iteration is ', self.bst.best_iteration)


  def train_with_total(self, iteratoion=None):
    train_total = pd.concat([self.train_set, self.valid_set])
    assert isinstance(train_total, pd.DataFrame)

    label = train_total['answer_flag']
    train = train_total.drop(['question_id', 'user_id', 'answer_flag',
                                 'is_valid'], axis=1)

    valid = self.valid_set.drop(['question_id', 'user_id', 'answer_flag',
                                 'is_valid'], axis=1)

    dvalid = XGMatrix(data=valid, label=self.valid_set['answer_flag'])

    total_group = list()
    total_group.extend(self.train_set_group)
    total_group.extend(self.valid_set_group)
    dtrain = XGMatrix(data=train, label=label)

    dtrain.set_group(total_group)
    dvalid.set_group(self.valid_set_group)

    watchlist = [(dtrain, 'train'),
                 (dvalid, 'valid')]

    if iteratoion == None:
      best_iteration = self.bst.best_iteration
    else :
      best_iteration = iteratoion

    self.bst = xgb.train(self.conf['xgb_param'], dtrain, best_iteration, watchlist,
                         feval=self.ndcgerror)

  def predict(self):

    online_valid = self.submit_valid_set.drop(['qid', 'uid', 'label'], axis=1)
    dsubmit = XGMatrix(data=online_valid)
    predict = self.bst.predict(dsubmit, ntree_limit=self.bst.best_iteration)

    max = np.max(predict)
    min = np.min(predict)

    output = []



    for i in self.submit_valid_set.index:
      output.append([self.submit_valid_set['qid'][i],
                     self.submit_valid_set['uid'][i],
                     (predict[i] - min)/(max - min) ])
    #
    import csv
    #
    output_file = open('output/valid.csv', 'w')
    writer = csv.writer(output_file)
    writer.writerow(['qid', 'uid', 'label'])
    for x in output:
      writer.writerow([x[0], x[1], x[2]])

    output_file.close()



if __name__ == '__main__':
  import numpy as np
  try:
    version = sys.argv[1]
  except:
    version = '1'


  conf = {
    'sample': 0,
    'version': version,

    'training_params': {
      'save_every': 1,
      'batch_size': 256,
      'nb_epoch': 5000,
      'validation_split': 0.1,
    },
    'xgb_param' : {'max_depth': 6,
             'eta': 0.001,
             'silent': 1,
             # 'gamma': 0,
             'min_child_weight': 5,
             # 'objective': 'reg:linear',
             'objective': 'rank:pairwise',
             # 'objective': 'binary:logistic',
             # 'num_class' : 2,
             'subsample': 0.9,
             'booster': 'gbtree',
             # 'eval_metric': ['ndcg@5', 'ndcg@10'],
             # 'booster': 'gblinear',
             # 'alpha': 0.001, 'lambda': 1,
             # 'subsample': 0.5
             }

  }

  evaluator = Evaluator(conf)
  # evaluator.train()
  evaluator.train_with_total(917)
  evaluator.predict()

