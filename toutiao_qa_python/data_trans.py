# coding=utf-8
try:
  import cPickle as pickle
except:
  import pickle
import pandas as pd

question_info = pd.read_csv('question_info.txt', sep='\t', header=None,
            names=[
                'question_id',      #c1c0075239841777d5b01c40b38135d2
                'question_tag',     #0
                'words_seq',        #1,2,3,4,5
                'character_seq',    #1,2,3,4,5
                'point_num',        #123
                'reply_num',        #321
                'good_reply_num'    #99
            ], index_col='question_id')

import numpy as np

question_info['words_seq'] = \
    [[] if x != x or x == '/' else list(np.array(x.split('/'), dtype=int)) for
     x in
        question_info['words_seq']]

question_info['character_seq'] = \
    [[] if x != x or x == '/' else list(np.array(x.split('/'), dtype=int)) for
     x in
        question_info['character_seq']]


user_info = pd.read_csv('user_info.txt', sep='\t', header=None,
            names=[
                'user_id',      #c1c0075239841777d5b01c40b38135d2
                'user_tags',     #0,1,2
                'user_desc_words_sec',        #1,2,3,4,5
                'user_desc_characters_sec',    #1,2,3,4,5
            ], index_col='user_id')

user_info['user_desc_words_sec'] = \
    [[] if x != x or x == '/' else list(np.array(x.split('/'), dtype=int))
     for x in user_info['user_desc_words_sec']]

user_info['user_desc_characters_sec'] = \
    [[] if x != x or x == '/' else list(np.array(x.split('/'), dtype=int)) for
        x in
        user_info['user_desc_characters_sec']]

invited_info_train = pd.read_csv('invited_info_train.txt', sep='\t',
                                 header=None,
                                 names=[
                                     'question_id',
                                     'user_id',
                                     'answer_flag',        #0/1
                                 ])

invited_info_valid = pd.read_csv('invited_info_validate_nolabel.txt', sep='\t',
                                 header=None,
                                 names=[
                                     'question_id',
                                     'user_id',
                                 ])


dictionary = pd.concat([
    question_info['words_seq'],
    user_info['user_desc_words_sec']],
    ignore_index=True)

w = []  # 将所有词语整合在一起
for i in dictionary:
  w.extend(i)

dictionary = pd.DataFrame(pd.Series(w).value_counts())  # 统计词的出现次数

pickle.dump(dictionary, open('pkl/dictionary.pkl', 'wb'))
pickle.dump(question_info, open('pkl/question_info.pkl', 'wb'))
pickle.dump(user_info, open('pkl/user_info.pkl', 'wb'))
pickle.dump(invited_info_train, open('pkl/invited_info_train.pkl', 'wb'))
pickle.dump(invited_info_valid, open('pkl/invited_info_valid.pkl', 'wb'))

