import pickle
from gensim.models import Word2Vec

def load(name):
  return pickle.load(open('toutiao_qa_python/pkl/%s' % name, 'rb'))


question_info = load('question_info.pkl')
user_info = load('user_info.pkl')

setences = []
# for x in question_info['character_seq']:
#   setences.extend(x)
# for x in user_info['user_desc_words_sec']:
#   setences.extend(x)

import numpy as np

setences.extend([list(np.array(x, dtype=str)) for x in question_info['words_seq'] if len(x)>0])
setences.extend([list(np.array(x, dtype=str)) for x in user_info['user_desc_words_sec'] if len(x)>0])

size = 100

w2v = Word2Vec(setences, size=size, window=5, min_count=0)

#
w2v.save('w2v_word_embending_%d.m' % size)

setences = []
setences.extend([list(np.array(x, dtype=str)) for x in question_info['character_seq'] if len(x)>0])
setences.extend([list(np.array(x, dtype=str)) for x in user_info['user_desc_characters_sec'] if len(x)>0])

w2v = Word2Vec(setences, size=size, window=10, min_count=0)
w2v.save('w2v_character_embending_%d.m' % size)

setences = []
setences.extend([x.split('/') for x in user_info['user_tags']])

setences.extend([[str(x)] for x in question_info['question_tag']])

w2v = Word2Vec(setences, size=size, window=5, min_count=0)
w2v.save('w2v_tag_embending_%d.m' % size)

