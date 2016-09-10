from __future__ import print_function

import os
import pandas as pd
import sys
import random
from time import strftime, gmtime, localtime

import pickle

from keras.optimizers import Adam
from keras_models_w2v import AttentionModelW2V

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
    self.w2v_len = conf.get('w2v_len')
    self.w2v = self.load_w2v()


  ##### Resources #####

  def load(self, name):
    return pickle.load(open(os.path.join(self.path, name), 'rb'))

  ##### Loading / saving #####

  def save_model(self, model):
    if not os.path.exists('models/'):
      os.makedirs('models/')
    if not os.path.exists('models/%s' % self.conf.get('model_dir')):
      os.makedirs('models/%s' % self.conf.get('model_dir'))

    pickle.dump(self.conf, open('models/%s/conf' % self.conf.get('model_dir'),
                                'wb'))
    model.save_model('models/%s/model.json' % self.conf.get('model_dir'))

  def load_model(self, model_dir):
    assert os.path.exists('models/%s/model.json' % model_dir),\
      'model file not found'
    model = AttentionModelW2V(self.conf)
    return model.load_model('models/%s/model.json' % model_dir)

  def save_epoch(self, model, epoch):
    if not os.path.exists('models/'):
      os.makedirs('models/')
    if not os.path.exists('models/%s' % self.conf.get('model_dir')):
      os.makedirs('models/%s' % self.conf.get('model_dir'))
    model.save_weights('models/%s/weights_epoch_%d.h5' %
                       (self.conf.get('model_dir'), epoch), overwrite=True)

  def load_epoch(self, model, epoch):
    assert os.path.exists('models/%s/weights_epoch_%d.h5' %
                          (self.conf.get('model_dir'), epoch)), \
      'Weights at epoch %d not found' % epoch
    model.load_weights('models/%s/weights_epoch_%d.h5' %
                          (self.conf.get('model_dir'), epoch))

  ##### Padding #####

  def padq(self, data):
    return self.pad(data, self.conf.get('question_len', None))

  def pada(self, data):
    return self.pad(data, self.conf.get('answer_len', None))

  def pad(self, data, len=None):
    from keras.preprocessing.sequence import pad_sequences
    return pad_sequences(data, maxlen=len, padding='post',
                         truncating='post', value=-1)

  ##### Training #####

  @staticmethod
  def print_time():
    print(strftime('%Y-%m-%d %H:%M:%S :: ', localtime()), end='')

  def train(self, model):
    save_every = self.params.get('save_every', None)
    batch_size = self.params.get('batch_size', 128)
    nb_epoch = self.params.get('nb_epoch', 10)
    split = self.params.get('validation_split', 0)

    bad_answer_sample = self.params.get('bad_answer_sample', 0)

    training_set = self.load('invited_info_train.pkl')
    question_info = self.load('question_info.pkl')
    user_info = self.load('user_info.pkl')

    print('start sequences padding')

    question_info['words_seq_padding'] = list(self.padq(list(question_info[
                                                          'words_seq'])))
    user_info['user_desc_words_sec_padding'] = list(self.pada(list(user_info[
                                                              'user_desc_words_sec'])))
    print('start word2vec mapping')

    get_w2v = lambda w: [
      self.w2v[str(x)] if self.w2v.__contains__(str(x)) else np.zeros(
              self.w2v_len)
      for x in w]

    question_info['words_seq_padding_w2v'] = \
      question_info['words_seq_padding'].apply(get_w2v)
    user_info['user_desc_words_sec_padding_w2v'] = \
      user_info['user_desc_words_sec_padding'].apply(get_w2v)

    # questions = list()
    # answers = list()

    train_group = training_set.groupby('question_id')

    all_users = list(user_info.index)

    question_ids = list()
    good_answer_ids = list()
    bad_answer_ids = list()

    questions = [x[0] for x in list(train_group)]
    len_split = int(len(questions) * split)
    valid_questions = random.sample(questions, len_split)

    valid_qid = list()
    valid_uid = list()
    valid_answer = list()

    print('start sampling')

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

    print('start seq mapping')
    print('step 1, question seq')
    question_words_seq = np.array([
        question_info['words_seq_padding_w2v'][x]
        for x in question_ids])

    print('step 2, answer good seq')
    answers_good_words_seq = np.array([
        user_info['user_desc_words_sec_padding_w2v'][x]
        for x in good_answer_ids])

    print('step 3, answer bad seq')
    answers_bad_words_seq = np.array([
        user_info['user_desc_words_sec_padding_w2v'][x]
        for x in bad_answer_ids])

    print('start valid data set generater')
    # valid set
    valid_question_words_seq = np.array([
      question_info['words_seq_padding_w2v'][x]
      for x in valid_qid])

    valid_answers_words_seq = np.array([
      user_info['user_desc_words_sec_padding_w2v'][x]
      for x in valid_uid])

    valid_data = {'qid': valid_qid, 'uid': valid_uid, 'answer_flag':
      valid_answer}
    valid_set = pd.DataFrame(data=valid_data)


    # val_loss = {'loss': 1., 'epoch': 0}
    val_ndcg = {'ndcg':0, 'epoch':0}

    print('saving model')
    self.save_model(model)
    print('start training')

    for i in range(1, nb_epoch):
      # sample from all answers to get bad answers
      print('Epoch %d :: ' % i, end='')
      self.print_time()
      hist = model.fit([question_words_seq, answers_good_words_seq,
                       answers_bad_words_seq],
                       nb_epoch=1,
                       batch_size=batch_size,
                       # validation_split=split
                       )

      predict = model.prediction_model.predict(
              [valid_question_words_seq, valid_answers_words_seq],
              batch_size=batch_size, verbose=1)

      valid_set['predict'] = [x[0][0] for x in predict]

      valid_group = valid_set.groupby('qid')

      scores = list()

      for x in list(valid_group):
        # question_id = x[0]
        answer_info = x[1].sort('predict', ascending=False)
        predict = [answer_info['predict'][x] * answer_info['answer_flag'][x]
                   for x in answer_info.index]
        from ndcg import ndcg_at_k
        scores.append(
          ndcg_at_k(predict, 5) * 0.5 + ndcg_at_k(predict, 10) * 0.5)

      valid_ndcg = np.mean(scores)
      print('ndcg mean is %lf' % valid_ndcg)
      if valid_ndcg > val_ndcg['ndcg']:
        val_ndcg = {'ndcg': valid_ndcg, 'epoch':i}

      print('Best: Ndcg = {}, Epoch = {}'.format(val_ndcg['ndcg'],
                                                 val_ndcg['epoch']))

      # if hist.history['val_loss'][0] < val_loss['loss']:
      #   val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
      # print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'],
      #                                            val_loss['epoch']))

      if save_every is not None and i % save_every == 0:
        self.save_epoch(model, i)

    # return val_loss
    return val_ndcg

  def load_w2v(self):
    from gensim.models import Word2Vec
    return Word2Vec.load('w2v_embending_%d.m' % self.w2v_len)


if __name__ == '__main__':
  import numpy as np
  try:
    model_dir = sys.argv[1]
  except:
    model_dir = '.'

  print('model path is ', model_dir)

  conf = {
    'question_len': 20,
    'answer_len': 20,
    # 'w2v_len': 256,
    'w2v_len': 50,
    # 'margin': 0.02,
    'margin': 0.05,
    # 'margin': 0.5,
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
      'bad_answer_sample': 1,
    },

    'model_params': {
      # 'n_embed_dims': 100,
      # 'n_embed_dims': 256,
      # 'n_embed_dims': 128,
      # 'n_hidden': 200,

      # convolution
      # 'nb_filters': 1000,
      # * 4
      # 'conv_activation': 'tanh',

      # recurrent
      # 'n_lstm_dims': 141,
      'n_lstm_dims': 64,
      # * 2

      # 'initial_embed_weights':
      #   np.load('./word2vec_100_dim.embeddings'),
      'similarity_dropout': 0.5,
    },

    'similarity_params': {
      'mode': 'gesd',
      # 'mode': 'aesd',
      'gamma': 1,
      'c': 1,
      'd': 2,
    }
  }

  evaluator = Evaluator(conf)

  ##### Define model ######
  model = AttentionModelW2V(conf)
  optimizer = conf.get('training_params', dict()).get('optimizer', 'rmsprop')
  model.compile(optimizer=optimizer,
                # metrics=['accuracy']
                )

  best_loss = evaluator.train(model)

