import json
import os
import numpy as np

glove_dim = 300
glove_vector = 'data/glove.6B.%dd.txt' %glove_dim

def word_tokenize(s):
  sent = s.lower()
  sent = re.sub('[^A-Za-z0-9\s]+',' ', sent)
  return sent.split()

def read_json(json_file): 
  with open(json_file) as data_file: 
    data = json.load(data_file)
  return data 

def save_json(data, json_file):
  with open(json_file, 'w') as f:
    json.dump(data, f)
  print "Dumped json file to %s." %json_file

class glove_embedding(object):

  def __init__(self, glove_file=glove_vector):
    glove_txt = open(glove_file).readlines()
    glove_txt = [g.strip() for g in glove_txt]
    glove_vector = [g.split(' ') for g in glove_txt]
    glove_words = [g[0] for g in glove_vector]
    glove_vecs = [g[1:] for g in glove_vector]
    glove_array = np.zeros((glove_dim, len(glove_words)))
    glove_dict = {}
    for i, w in enumerate(glove_words):  glove_dict[w] = i
    for i, vec in enumerate(glove_vecs):
      glove_array[:,i] = np.array(vec)
    self.glove_array = glove_array
    self.glove_dict = glove_dict
    self.glove_words = glove_words

  def cosine_similarity(self, word):
    word_vec = self.glove_array[:,self.glove_dict[word]]
    cosine_sim_numerator = np.dot(word_vec, self.glove_array)
    cosine_sim_denominator = np.linalg.norm(word_vec)*np.linalg.norm(self.glove_array, axis=0)
    return cosine_sim_numerator/cosine_sim_denominator

