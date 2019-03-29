import sys
import os
import numpy as np
import copy
sys.path.insert(0, 'caffe/python/')
import caffe
import random
import pickle as pkl
import caffe
from multiprocessing import Pool
from threading import Thread
import random
import h5py
import itertools
import math
import re
from python_utils import *

glove_dim = 300
glove_path = 'data/glove.6B.%dd.txt' %glove_dim
debug = False 
if debug:
    glove_path = 'data/glove_debug_path.txt'

#Glove takes a really time to load so when I am debugging my code I have a smaller text file; 
#I have a break point to make sure I never accidentally ran experiments with my debug txt file.
if glove_path == 'data/glove_debug_path.txt':
    print "In debug mode: continue?"
    import pdb; pdb.set_trace()

n_clusters = 3000
length_prep_word = 40
length_prep_character = 250

vocab_file = 'data/vocab_glove_complete.txt'

"""
Etc. functions
"""

def word_tokenize(s):
  sent = s.lower()
  sent = re.sub('[^A-Za-z0-9\s]+',' ', sent)
  return sent.split()

def sentences_to_words(sentences):
  words = []
  for s in sentences:
    words.extend(word_tokenize(str(s.lower())))
  return words

def language_process_labels(label_hash_file):
    label_hash = pkl.load(open(label_hash_file, 'r'))
    labels = list(set(label_hash.values()))
    print "Reading glove embedding"
    embedding = glove_embedding(glove_path)

    label_dict = {}
    for label in labels:
        label_mod = label.split(' (')[0]
        label_words = word_tokenize(label_mod)
        label_input = np.zeros((len(label_words), glove_dim))
        for i, word in enumerate(label_words):
            try:
                label_input[i,:] = embedding.glove_array[:,embedding.glove_dict[word]]
            except:
                print "%s is not in embedding" %word
        label_dict[label] = label_input
    return label_dict 

"""
General functions to process language.
"""

class recurrent_language(object):

  def get_vocab_size(self):
    return len(self.vocab_dict.keys()) 
    
  def preprocess(self, data):

    vector_dim = self.get_vector_dim()
    
    for d in data:
      words = sentences_to_words([d['description']])
      sentence_mat = np.zeros((len(words), vector_dim))
      count_words = 0
      for i, w in enumerate(words):
        try:
          sentence_mat[count_words,:] = self.vocab_dict[w]
          count_words += 1
        except:
          if '<unk>' in self.vocab_dict.keys():
            sentence_mat[count_words,:] = self.vocab_dict['<unk>'] 
            count_words += 1
          else:
            pass
      sentence_mat = sentence_mat[:count_words] 
      d['language_input'] = sentence_mat
      d['gt'] = (d['gt'][0], d['gt'][1]) #hacky preprocessing thing; makes list into tuple
    return data

class recurrent_embedding(recurrent_language):

  def read_embedding(self):
    print "Reading glove embedding"
    embedding = glove_embedding(glove_path)
    self.embedding = embedding

  def get_vector_dim(self):
    return glove_dim 

  def __init__(self, data):

    self.read_embedding()
    embedding = self.embedding
    vector_dim = self.get_vector_dim()
    self.data = data

    self.data = data
    vocab = open(vocab_file).readlines()
    vocab = [v.strip() for v in vocab] 
    if '<unk>' in vocab: 
      vocab.remove('<unk>') #don't have an <unk> vector.  Alternatively, could map to random vector...
    vocab_dict = {}

    for i, word in enumerate(vocab):
      try:
        vocab_dict[word] = embedding.glove_array[:,embedding.glove_dict[word]] 
      except:
        print "%s not in glove embedding" %word
    self.vocab_dict = vocab_dict

  def preprocess(self, data):

    vector_dim = self.get_vector_dim()
    
    for d in data:
      d['language_input'] = sentences_to_words([d['description']])

    return data
    
  def get_vocab_dict(self):
    return self.vocab_dict

"""
General visual feature processing functions.
"""

def feature_process_base(start, end, features):
  return np.mean(features[start:end+1,:], axis = 0)

def feature_process_norm(start, end, features):
  base_feature = np.mean(features[start:end+1,:], axis = 0)
  return base_feature/(np.linalg.norm(base_feature) + 0.00001)

def feature_process_context(start, end, features):
  feature_dim = features.shape[1]
  full_feature = np.zeros((feature_dim*2,))
  if np.sum(features[5,:]) > 0:
    full_feature[:feature_dim] = feature_process_norm(0,6, features) 
  else:
    full_feature[:feature_dim] = feature_process_norm(0,5, features) 
  full_feature[feature_dim:feature_dim*2] = feature_process_norm(start, end, features) 

  return full_feature

def feature_process_before_after(start, end, features):
    feature_dim = features.shape[1]
    full_feature = np.zeros((feature_dim*3,))
    if start > 0:
      full_feature[:feature_dim] = feature_process_norm(start-1, start-1, features) 
    full_feature[feature_dim:feature_dim*2] = feature_process_norm(start, end, features) 
    if end < 5:
      full_feature[feature_dim*2:feature_dim*3] = feature_process_norm(end+1, end+1, features) 
 
    return full_feature

feature_process_dict = {'feature_process_norm': feature_process_norm,
                        'feature_process_context': feature_process_context,
                        'feature_process_before_after': feature_process_before_after,
                        }
   
class extractData(object):
  """ General class to iterate over data
  """

  def increment(self): 
    next_batch = [None]*self.batch_size
    if self.iteration + self.batch_size >= self.num_data:
      next_batch[:self.num_data-self.iteration] = self.data_list[self.iteration:]
      next_batch[self.num_data-self.iteration:] = self.data_list[:self.batch_size -(self.num_data-self.iteration)]
      random.shuffle(self.data_list)
      self.iteration = self.num_data - self.iteration
    else:
      next_batch = self.data_list[self.iteration:self.iteration+self.batch_size]
      self.iteration += self.batch_size
    assert self.iteration > -1
    assert len(next_batch) == self.batch_size 
    return next_batch
 
  def advanceBatch(self, increment=True):
    if increment:
      next_batch = self.increment()
      self.next_batch = next_batch
    self.get_data(self.next_batch)


class extractRecurrentLanguageFeaturesEfficient(extractData):

  def __init__(self, dataset, params, result=None):
    self.data_list = range(len(dataset))
    self.num_data = len(self.data_list)
    self.dataset = dataset
    self.iteration = 0

    self.vocab_dict = params['vocab_dict']
    self.batch_size = params['batch_size']
    self.num_glove_centroids = self.vocab_dict.values()[0].shape[0] 
    self.T = params['sentence_length']

    if isinstance(result, dict):
        self.result = result
        self.bog_key = params['bog_key']
        self.cont_key = params['cont_key']
    
        self.top_keys = [self.bog_key, self.cont_key]
        self.top_shapes = [(self.T, self.batch_size, self.num_glove_centroids),
                           (self.T, self.batch_size)]
    else:
        print "Will only be able to run in test mode"

  def get_features(self, query):

    feature = np.zeros((self.T, self.num_glove_centroids)) 
    cont = np.zeros((self.T,)) 

    len_query = min(len(query), self.T)
    if len_query < len(query):
      query = query[:len_query]
    for count_word, word in enumerate(query):
      try:
        feature[-(len_query)+count_word,:] = self.vocab_dict[word] 
      except:
        feature[-(len_query)+count_word,:] = np.zeros((glove_dim,))
    cont[-(len_query-1):] = 1 
    assert np.sum(feature[:-len_query,:]) == 0

    return feature, cont

  def get_data_test(self, data):
    query = data['language_input']
    return self.get_features(query) 

  def get_data(self, next_batch):

    data = self.dataset
    bog = np.zeros((self.T, self.batch_size, self.num_glove_centroids))
    cont = np.zeros((self.T, self.batch_size))

    for i, nb in enumerate(next_batch):
      query = data[nb]['language_input']
      bog[:,i,:], cont[:,i] = self.get_features(query)


    self.result[self.bog_key] = bog 
    self.result[self.cont_key] = cont 

class extractRelationalClipFeatures(extractData):
  
  def __init__(self, dataset, params, result):
    self.data_list = range(len(dataset))
    self.feature_process_algo = params['feature_process']
    self.loc_feature = params['loc_feature']
    self.num_data = len(self.data_list)
    self.dataset = dataset
    self.iteration = 0
    self.internal_iteration = 0
    self.loc = params['loc_feature']
    self.strong_supervision = params['strong_supervision']
    self.params = params
    self.supervise_lw = 1.

    loss_type = params['loss_type']
    assert loss_type in ['triplet', 'inter', 'intra']

    self.inter = False
    self.intra = False
    if loss_type in ['triplet', 'inter']:
      self.inter = True
    if loss_type in ['triplet', 'intra']:
      self.intra = True

    self.batch_size = params['batch_size']
    self.num_glove_centroids = params['num_glove_centroids']

    features_h5py = h5py.File(params['features'])
    features = {}
    for key in features_h5py.keys():
      features[key] = np.array(features_h5py[key])
    features_h5py.close()
    self.features = features

    assert self.feature_process_algo == 'feature_process_norm' 

    self.feature_process = feature_process_dict[self.feature_process_algo]

    self.feature_dim = self.feature_process(0,0,self.features[self.dataset[0]['video']]).shape[-1]
    self.feature_dim += 2 
    self.result = result

    L = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5)]
    for i in itertools.combinations(range(6), 2):
      L.append(i)
    self.possible_annotations = L

    self.possible_annotations_dict = {}
    for i, item in enumerate(self.possible_annotations):
        self.possible_annotations_dict[item] = i

    self.feature_p = 'features_p' 
    self.feature_n = 'features_n' 
    self.feature_global_p = 'features_global_p'

    self.top_keys = [self.feature_p, self.feature_n, self.feature_global_p]
    self.top_shapes = [(self.batch_size, 1, self.feature_dim),
                     (self.batch_size, 1, self.feature_dim),
                     (self.batch_size, len(self.possible_annotations), self.feature_dim)]

    if self.inter:
      self.feature_inter = 'features_inter'
      self.feature_global_inter = 'features_global_inter'
      self.top_keys.append(self.feature_inter)
      self.top_keys.append(self.feature_global_inter)
      self.top_shapes.append((self.batch_size, 1, self.feature_dim)) 
      self.top_shapes.append((self.batch_size, len(self.possible_annotations), self.feature_dim)) 
    if self.strong_supervision:
      self.top_keys.append('strong_supervision_loss')
      self.top_shapes.append((self.batch_size,1))  
 
  def get_features(self, video, endpoints): 
    if (isinstance(video, str)) or (isinstance(video, unicode)):
        features = self.features[video]
    else:
        features = video
    full_feature = np.zeros((self.feature_dim,))
    full_feature[:-2] = self.feature_process(endpoints[0], endpoints[1], features)

    if self.loc:
        full_feature[-2] = endpoints[0]/6.
        full_feature[-1] = endpoints[1]/6.

    return full_feature 

  def get_data_test(self, data):

    video = data['video']
    video_features = np.zeros((len(self.possible_annotations), self.feature_dim))

    for count_pa, pa in enumerate(self.possible_annotations):
      video_features[count_pa,:] = self.get_features(video, [pa[0], pa[1]])

    return video_features

  def get_data(self, next_batch):
    self.internal_iteration += 1
    feature_process = self.feature_process
    data = self.dataset
    features_p = np.zeros((self.batch_size, 1, self.feature_dim))
    features_n = np.zeros((self.batch_size, 1, self.feature_dim))
    features_global_p = np.zeros((self.batch_size, len(self.possible_annotations), self.feature_dim))
    if self.inter: 
      features_inter = np.zeros((self.batch_size, 1, self.feature_dim))
      features_global_inter = np.zeros((self.batch_size, len(self.possible_annotations), self.feature_dim))
    if self.strong_supervision:
      strong_supervision_loss = np.zeros((self.batch_size,1))  

    for i, nb in enumerate(next_batch):

      # get start/end points
      rint = 0
      train_times = [tuple(t) for t in data[nb]['train_times']]
      if len(train_times) > 0:
          rint = random.randint(0,len(data[nb]['train_times'])-1)
      gt_s = train_times[rint][0]
      gt_e = train_times[rint][1]
      
      video = data[nb]['video']
      feats = self.features[video]
      
      possible_n = list(set(self.possible_annotations) - set([(gt_s, gt_e)])) 
      random.shuffle(possible_n)
      n = possible_n[0]
      assert n != (gt_s, gt_e) 
      
      if self.inter:
        other_video = data[nb]['video']
        while (other_video == video):
          other_video_index = int(random.random()*len(data))
          other_video = data[other_video_index]['video'] 
        feats_inter = self.features[other_video]
      
      features_p[i,0,:] = self.get_features(feats, [gt_s, gt_e])
      features_n[i,0,:] = self.get_features(feats, n)
      if self.inter:
         features_inter[i,0,:] = self.get_features(other_video, [gt_s, gt_e])
    
      for count_pa, pa in enumerate(self.possible_annotations):
        features_global_p[i,count_pa,:] = self.get_features(feats, [pa[0], pa[1]])
      if self.inter:
        for count_pa, pa in enumerate(self.possible_annotations):
          features_global_inter[i,count_pa,:] = self.get_features(other_video, [pa[0], pa[1]])
      if self.strong_supervision:
        if (self.internal_iteration > 0) & (self.internal_iteration % (self.num_data/self.batch_size) == 0):
            if self.params['decay_context_supervision']: #didn't end up using this in the end, but it seems like this could be helpful!
                self.supervise_lw = self.supervise_lw*0.95
    
        """
        For some video moments we might not have the ground truth context.  E.g., for the original videos from DiDeMo.
        Here we check to see if there is context stored in the data, and if so use it.
        Additionally, we can force the context moment to be the global context moment from MCN if the gt context is not known.
        I played around with this (since the global context seemed to be helpful on original DiDeMo), but it did not help much. 
        """
        if len(data[nb]['context']) > 0: 
            #Copy context to first context slot for strong supervision
            context = data[nb]['context']
            idx = self.possible_annotations_dict[tuple(context)]
            idx0 = features_global_p[i,0,:].copy() 
            features_global_p[i,0,:] = features_global_p[i,idx,:].copy()
            features_global_p[i,idx,:] = idx0 
            strong_supervision_loss[i] = self.supervise_lw 
        elif self.params['global_supervision']:
            #if we do not know gt supervision, uses global supervision by default
            idx = self.possible_annotations_dict[(0,5)]
            idx0 = features_global_p[i,0,:].copy() 
            features_global_p[i,0,:] = features_global_p[i,idx,:].copy()
            features_global_p[i,idx,:] = idx0 
            strong_supervision_loss[i] = self.supervise_lw

    assert not math.isnan(np.mean(features_p))

    if self.inter:
      assert not math.isnan(np.mean(features_inter))
      assert not math.isnan(np.mean(features_global_inter))

    #This is less than ideal in terms of code neatness.  I create context TEF features in "build_net.py" by subtracting the global feature from the moment feature.  To create a model with no tef, I just zero our the tef for the global features.  In effect, this just means that the moment TEF features will be replicated.
    if not self.params['context_tef']:
      features_global_p[:,:,-2:] = 0
      features_global_inter[:,:,-2:] = 0

    self.result[self.feature_p] = features_p
    self.result[self.feature_n] = features_n
    self.result[self.feature_global_p] = features_global_p
    
    if self.inter:
      self.result[self.feature_inter] = features_inter
      self.result[self.feature_global_inter] = features_global_inter
    if self.strong_supervision:
      self.result['strong_supervision_loss'] = strong_supervision_loss
 
class extractAverageClipFeatures(extractData):
  
  def __init__(self, dataset, params, result):
    self.data_list = range(len(dataset))
    self.feature_process_algo = params['feature_process']
    self.loc_feature = params['loc_feature']
    self.num_data = len(self.data_list)
    self.dataset = dataset
    self.iteration = 0
    self.loc = params['loc_feature']
    loss_type = params['loss_type']
    assert loss_type in ['triplet', 'inter', 'intra']

    self.inter = False
    self.intra = False
    if loss_type in ['triplet', 'inter']:
      self.inter = True
    if loss_type in ['triplet', 'intra']:
      self.intra = True

    self.batch_size = params['batch_size']
    self.num_glove_centroids = params['num_glove_centroids']

    features_h5py = h5py.File(params['features'])
    features = {}
    for key in features_h5py.keys():
      features[key] = np.array(features_h5py[key])
    features_h5py.close()
    self.features = features

    assert self.feature_process_algo in feature_process_dict.keys()
    self.feature_process = feature_process_dict[self.feature_process_algo]

    self.feature_dim = self.feature_process(0,0,self.features[self.dataset[0]['video']]).shape[-1]
    self.result = result

    self.bog_key = params['bog_key']
    self.feature_key_p = 'features_p' 
    self.feature_time_stamp_p = 'features_time_stamp_p'
    self.feature_time_stamp_n = 'features_time_stamp_n'

    self.top_keys = [self.feature_key_p, self.feature_time_stamp_p, self.feature_time_stamp_n]
    self.top_shapes = [(self.batch_size, self.feature_dim),
                     (self.batch_size, 2),
                     (self.batch_size,2)]

    if self.inter:
      self.feature_key_inter = 'features_inter'
      self.top_keys.append(self.feature_key_inter)
      self.top_shapes.append((self.batch_size, self.feature_dim)) 
    if self.intra:
      self.feature_key_intra = 'features_intra'
      self.top_keys.append(self.feature_key_intra)
      self.top_shapes.append((self.batch_size, self.feature_dim)) 
    
    L = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5)]
    for i in itertools.combinations(range(6), 2):
      L.append(i)
    self.possible_annotations = L

  def get_data_test(self, d):
      video_feats = self.features[d['video']]
      features = np.zeros((len(self.possible_annotations), self.feature_dim))
      loc_feats = np.zeros((len(self.possible_annotations), 2))
      for i, p in enumerate(self.possible_annotations):
          features[i,:] = self.feature_process(p[0], p[1], video_feats)
          loc_feats[i,:] = [p[0]/6., p[1]/6.]

      return features, loc_feats

  def get_data(self, next_batch):

    feature_process = self.feature_process
    data = self.dataset
    features_p = np.zeros((self.batch_size, self.feature_dim))
    if self.inter: features_inter = np.zeros((self.batch_size, self.feature_dim))
    if self.intra: features_intra = np.zeros((self.batch_size, self.feature_dim))

    features_time_stamp_p = np.zeros((self.batch_size, 2))
    features_time_stamp_n = np.zeros((self.batch_size, 2))

    for i, nb in enumerate(next_batch):

      # get start/end points
      rint = 0
      train_times = [tuple(t) for t in data[nb]['train_times']]
      if len(train_times) > 0:
          rint = random.randint(0,len(train_times)-1)
      gt_s = train_times[rint][0]
      gt_e = train_times[rint][1]
  
      possible_n = list(set(self.possible_annotations) - set([(gt_s, gt_e)])) 
      random.shuffle(possible_n)
      n = possible_n[0]
      assert n != (gt_s, gt_e) 
     
      video = data[nb]['video']
      feats = self.features[video]
    
      if self.inter:
        other_video = data[nb]['video']
        while (other_video == video):
          other_video_index = int(random.random()*len(data))
          other_video = data[other_video_index]['video'] 
        feats_inter = self.features[other_video]
     
      features_p[i,:] = feature_process(gt_s, gt_e, feats)
      if self.intra:
        features_intra[i,:] = feature_process(n[0], n[1], feats)
      if self.inter:
        features_inter[i,:] = feature_process(gt_s, gt_e, feats_inter)
    
      if self.loc:
        features_time_stamp_p[i,0] = gt_s/6.
        features_time_stamp_p[i,1] = gt_e/6.
        features_time_stamp_n[i,0] = n[0]/6.
        features_time_stamp_n[i,1] = n[1]/6.
      else:
        features_time_stamp_p[i,0] = 0 
        features_time_stamp_p[i,1] = 0
        features_time_stamp_n[i,0] = 0
        features_time_stamp_n[i,1] = 0
 
      assert not math.isnan(np.mean(self.features[data[nb]['video']][n[0]:n[1]+1,:]))
      assert not math.isnan(np.mean(self.features[data[nb]['video']][gt_s:gt_e+1,:]))

    self.result[self.feature_key_p] = features_p
    self.result[self.feature_time_stamp_p] = features_time_stamp_p
    self.result[self.feature_time_stamp_n] = features_time_stamp_n
    if self.inter:
      self.result[self.feature_key_inter] = features_inter
    if self.intra:
      self.result[self.feature_key_intra] = features_intra

class batchAdvancer(object):
  
  def __init__(self, extractors):
    self.extractors = extractors
    self.increment_extractor = extractors[0]

  def __call__(self):
    #The batch advancer just calls each extractor
    next_batch = self.increment_extractor.increment()
    for e in self.extractors:
      e.get_data(next_batch)

class python_data_layer(caffe.Layer):
  """ General class to extract data.
  """

  def setup(self, bottom, top):
    self.params = eval(self.param_str)
    random.seed(self.params['python_random_seed'])
    params = self.params

    assert 'top_names' in params.keys()

    #set up prefetching
    self.thread_result = {}
    self.thread = None

    self.setup_extractors()
 
    self.batch_advancer = batchAdvancer(self.data_extractors) 
    shape_dict = {}
    self.top_names = []
    for de in self.data_extractors:
      for top_name, top_shape in zip(de.top_keys, de.top_shapes):
        shape_dict[top_name] = top_shape 
        self.top_names.append((params['top_names'].index(top_name), top_name)) 
    self.dispatch_worker()

    self.top_shapes = [shape_dict[tn[1]] for tn in self.top_names]

    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    self.join_worker()

    top_count = 0
    for top_index, name in self.top_names:
      shape = self.top_shapes[top_count] 
      print 'Top name %s has shape %s.' %(name, shape)
      top[top_index].reshape(*shape)
      top_count += 1

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
    if self.thread is not None:
      self.join_worker()

    for top_index, name in self.top_names:
      top[top_index].data[...] = self.thread_result[name]

    self.dispatch_worker()

  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_advancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None

  def backward(self, top, propoagate_down, bottom):
    pass

class dataLayer_ExtractPairedLanguageVision(python_data_layer):
 
  def setup_extractors(self):
    assert 'top_names' in self.params.keys()
    assert 'descriptions' in self.params.keys()
    assert 'features' in self.params.keys()
    if 'batch_size' not in self.params.keys(): self.params['batch_size'] = 120

    self.params['bog_key'] = 'BoG'
    if 'inputVisualData' not in self.params.keys():
       self.params['inputVisualData'] = 'clip'
    assert self.params['inputVisualData'] in ['clip', 'relational']

    language_extractor_fcn = extractRecurrentLanguageFeaturesEfficient
    self.params['cont_key'] = 'cont'

    if self.params['inputVisualData'] == 'clip':
      visual_extractor_fcn = extractAverageClipFeatures
    elif self.params['inputVisualData'] == 'relational':
      visual_extractor_fcn = extractRelationalClipFeatures
    else:  
      raise Exception("Did not indicate correct type of visual data") 

    language_process = recurrent_embedding
    data_orig = read_json(self.params['descriptions'])
    language_processor = language_process(data_orig)
    data = language_processor.preprocess(data_orig)
    self.params['vocab_dict'] = language_processor.vocab_dict
    num_glove_centroids = language_processor.get_vector_dim()
    self.params['num_glove_centroids'] = num_glove_centroids
    visual_feature_extractor = visual_extractor_fcn(data, self.params, self.thread_result)
    textual_feature_extractor = language_extractor_fcn(data, self.params, self.thread_result)
    self.data_extractors = [visual_feature_extractor, textual_feature_extractor]
