from __future__ import print_function
import sys
import os
sys.path.insert(0, 'caffe/python/')
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import argparse
import h5py
import numpy as np
from test_network import *
from data_processing import *
from python_utils import *
caffe.set_mode_gpu()

language_feature_process_dict = {'recurrent_embedding': recurrent_embedding}
recurrent_layers = ['lstm_no_embed']


def add_dict_values(key, my_dict):
  if my_dict.values():
    max_value = max(my_dict.values())
    my_dict[key] = max_value + 1
  else:
    my_dict[key] = 0
  return my_dict

pooling_type = {'max': 0, 'average': 1}

class retrieval_net(object):

  def pool_distances(self, vec, minimum_distance=True):
    #want to MINIMIZE distance; negate, maximize, then negate (again)
    #Assume that scores are Nx21 size blob
    if args.pool_type in ['max', 'average']:
        prep_pool = L.Reshape(vec, shape=dict(dim=[self.batch_size, 1, 21, 1]))

        if minimum_distance:
            prep_pool = L.Power(prep_pool, scale=-1)
        max_pool = L.Pooling(prep_pool, pool=pooling_type[self.args.pool_type], kernel_h=21, kernel_w=1)
        pool = L.Reshape(max_pool, shape=dict(dim=[self.batch_size]))
        if minimum_distance:
            pool = L.Power(pool, scale=-1)
    elif args.pool_type in ['sum']:
        #untested
        negative = L.Power(vec, scale=-1)
        pool = L.Reduction(negative, axis=1, operation=1) #sum
    else:
        raise Exception("You did not select a valid pooling type.")
    return pool

  def euclidean_distance(self, vec1, vec2, axis=1):
    negative = L.Power(vec2, scale=-1)
    difference = L.Eltwise(vec1, negative, operation=1)
    squared = L.Power(difference, power=2)
    reduction = L.Reduction(squared, axis=axis)
    return reduction
 
  def early_combine_mult_no_norm(self, vec1, vec2):
    eltwise = L.Eltwise(vec1, vec2, operation=P.Eltwise.PROD)
    setattr(self.n, 'feature', eltwise) 

    intermediate = L.InnerProduct(eltwise, num_output=self.visual_embedding_dim[-1], 
                           weight_filler=self.uniform_weight_filler(-0.08, .08), 
                           param=self.learning_params([[1,1], [2, 0]], ['eltwise_dist1', 'eltwise_dist1_b']), axis=1) 
    nonlin_1 = L.ReLU(intermediate)
    setattr(self.n, 'intermediate', nonlin_1) 
    dropout = L.Dropout(nonlin_1, dropout_ratio=self.dropout_visual)

    score = L.InnerProduct(dropout, num_output=1, 
                           weight_filler=self.uniform_weight_filler(-0.08, .08), 
                           param=self.learning_params([[1,1], [2, 0]], ['eltwise_dist2', 'eltwise_dist2_b']), axis=1) 
    negative_score = L.Power(score, scale=-1)
    setattr(self.n, 'rank_score', score) 
    return score

  def early_combine_mult_not_relational(self, vec1, vec2):
    mult = L.Eltwise(vec1, vec2, operation=0)
    setattr(self.n, 'mult', mult) 
    norm_mult = self.normalize(mult, numtiles=self.visual_embedding_dim[-1], axis=1)   
    setattr(self.n, 'norm_mult', norm_mult) 

    intermediate = L.InnerProduct(norm_mult, num_output=self.visual_embedding_dim[-1], 
                           weight_filler=self.uniform_weight_filler(-0.08, .08), 
                           param=self.learning_params([[1,1], [2, 0]], ['eltwise_dist1', 'eltwise_dist1_b']), axis=1) 
    nonlin_1 = L.ReLU(intermediate)
    setattr(self.n, 'intermediate', nonlin_1) 
    dropout = L.Dropout(nonlin_1, dropout_ratio=self.dropout_visual)

    score = L.InnerProduct(dropout, num_output=1, 
                           weight_filler=self.uniform_weight_filler(-0.08, .08), 
                           param=self.learning_params([[1,1], [2, 0]], ['eltwise_dist2', 'eltwise_dist2_b']), axis=1) 
    negative_score = L.Power(score, scale=-1)
    setattr(self.n, 'rank_score', score) 
    return score

  def early_combine_mult(self, vec1, vec2):
    mult = L.Eltwise(vec1, vec2, operation=0)
    setattr(self.n, 'mult', mult) 
    norm_mult = self.normalize(mult, numtiles=self.visual_embedding_dim[-1], axis=2)   
    setattr(self.n, 'norm_mult', norm_mult) 

    intermediate = L.InnerProduct(norm_mult, num_output=self.visual_embedding_dim[-1], 
                           weight_filler=self.uniform_weight_filler(-0.08, .08), 
                           param=self.learning_params([[1,1], [2, 0]], ['eltwise_dist1', 'eltwise_dist1_b']), axis=2) 
    nonlin_1 = L.ReLU(intermediate)
    setattr(self.n, 'intermediate', nonlin_1) 
    dropout = L.Dropout(nonlin_1, dropout_ratio=self.dropout_visual)

    score = L.InnerProduct(dropout, num_output=1, 
                           weight_filler=self.uniform_weight_filler(-0.08, .08), 
                           param=self.learning_params([[1,1], [2, 0]], ['eltwise_dist2', 'eltwise_dist2_b']), axis=2) 
    setattr(self.n, 'score', score) 
    negative_score = L.Power(score, scale=-1)
    setattr(self.n, 'negative_score', negative_score) 
    score = self.pool_distances(negative_score)
    setattr(self.n, 'pool_score', score) 
    return [score, negative_score]

  def tall_loss(self, positive, negative, query, lw=1):
    scores_p = self.distance_function(positive, query)
    scores_n = self.distance_function(negative, query)
    alpha_c = 1
    alpha_w = 1
    exp_p = L.Exp(scores_p, scale=-1)
    exp_n = L.Exp(scores_n)
    log_p = L.Log(exp_p, shift=1) 
    log_n = L.Log(exp_n, shift=1)
    scale_p = L.Power(log_p, scale=alpha_c) 
    scale_n = L.Power(log_n, scale=alpha_w) 
    all_scores = L.Concat(scale_p, scale_n, axis=0)
    return L.Reduction(all_scores, operation=4, loss_weight=[lw]) 

  def tall_feature(self, vec1, vec2, axis=1):

    #adddition
    add = L.Eltwise(vec1, vec2, operation=0)
    mult = L.Eltwise(vec1, vec2, operation=1)
    concat = L.Concat(vec1, vec2, axis=axis)
    concat = L.InnerProduct(concat, num_output=self.visual_embedding_dim[0], 
                           weight_filler=self.uniform_weight_filler(-0.08, .08), 
                           param=self.learning_params([[1,1], [2, 0]], ['tall_d', 'tall_db']), axis=axis) 

    self.n.tops['concat_feature'] = concat
    tall_feature = L.Concat(add, mult, concat, axis=axis)

    return tall_feature 

  def tall_distance(self, im_feature, language_feature):

    if self.args.tall_distance_size == 2:
        first_ip = self.visual_embedding_dim[0]
    else:
        first_ip = 1
    
    #make tall features
    feature = self.tall_feature(im_feature, language_feature)
    dropout = L.Dropout(feature, dropout_ratio=self.dropout_visual)
    distance =  L.InnerProduct(feature, num_output=first_ip, 
                               weight_filler=self.uniform_weight_filler(-0.08, .08),
                               bias_filler=self.constant_filler(0), 
                               param=self.learning_params([[1,1], [2,0]], ['tall_distance1', 'tall_distance_b1']), axis=1)
    if self.args.tall_distance_size == 2:
        nonlin_1 = L.ReLU(distance)
        dropout = L.Dropout(nonlin_1, dropout_ratio=self.dropout_visual)
        distance =  L.InnerProduct(dropout, num_output=1, 
                               weight_filler=self.uniform_weight_filler(-0.08, .08),
                               bias_filler=self.constant_filler(0), 
                               param=self.learning_params([[1,1], [2,0]], ['tall_distance2', 'tall_distance_b2']), axis=1)
    self.n.tops['rank_score'] = distance
    return distance 

  def early_combine_mult_tall(self, vec1, vec2):
    feature = self.tall_feature(vec1, vec2)
    setattr(self.n, 'feature', feature)
    intermediate = L.InnerProduct(feature, num_output=self.visual_embedding_dim[-1],
                                  weight_filler=self.uniform_weight_filler(-0.08, .08),
                                  param=self.learning_params([[1,1], [2, 0]], ['eltwise_dist1', 'eltwise_dist1_b']), axis=1)
    nonlin_1 = L.ReLU(intermediate)
    setattr(self.n, 'intermediate', nonlin_1)
    dropout = L.Dropout(nonlin_1, dropout_ratio=self.dropout_visual)

    score = L.InnerProduct(dropout, num_output=1,
                           weight_filler=self.uniform_weight_filler(-0.08, .08),
                           param=self.learning_params([[1,1], [2, 0]], ['eltwise_dist2', 'eltwise_dist2_b']), axis=1)
    negative_score = L.Power(score, scale=-1)
    setattr(self.n, 'rank_score', score)
    return score

  def __init__(self, args,
               data_layer='dataLayer_ExtractPairedLanguageVision', top_size=5,
               param_str = None, params={},
               is_test=False):
    self.n = caffe.NetSpec()
    self.silence_count = 0
    self.margin = args.margin
    self.is_test = is_test

    self.dropout_visual = args.dropout_visual
    self.dropout_language = args.dropout_language
    self.visual_embedding_dim = args.visual_embedding_dim 
    self.language_embedding_dim = args.language_embedding_dim 
    self.vision_layers = args.vision_layers
    self.language_layers = args.language_layers
    self.loc = args.loc
    self.data_layer = data_layer
    self.top_size = top_size
    self.param_str = param_str
    self.lw_inter = args.lw_inter
    self.lw_intra = args.lw_intra

    self.top_name_dict = params['top_names_dict']
    self.args = args
    self.T = params['sentence_length']
    self.count_im = 0
    self.local_unary_count = 0
    self.global_unary_count = 0

    self.inter = False
    self.intra = False
    if args.loss_type in ['triplet', 'inter']:
      self.inter = True
    if args.loss_type in ['triplet', 'intra']:
      self.intra = True

    assert self.inter or self.intra #need to have some type of loss!

    if 'batch_size' in param_str.keys():
      self.batch_size = param_str['batch_size']
    else:
      self.batch_size =120 
    self.params = params
    self.image_tag = args.image_tag

    if args.distance_function == 'tall_distance':
      self.distance_function = self.tall_distance
    elif args.distance_function == 'early_combine_mult_no_norm':
      self.distance_function = self.early_combine_mult_no_norm
    elif args.distance_function == 'early_combine_mult_tall':
      self.distance_function = self.early_combine_mult_tall
    elif args.distance_function == 'early_combine_mult' and args.input_visual_data == 'clip':
      self.distance_function = self.early_combine_mult_not_relational
    elif args.distance_function == 'early_combine_mult' and args.input_visual_data == 'relational':
      self.distance_function = self.early_combine_mult
    else:
      self.distance_function = self.euclidean_distance 

  def uniform_weight_filler(self, min_value, max_value):
    return dict(type='uniform', min=min_value, max=max_value)

  def constant_filler(self, value=0):
    return dict(type='constant', value=value)

  def learning_params(self, param_list, name_list = None):
    param_dicts = []
    for il, pl in enumerate(param_list):
      param_dict = {}
      param_dict['lr_mult'] = pl[0]
      if name_list:
        param_dict['name'] = name_list[il]
      if len(pl) > 1:
        param_dict['decay_mult'] = pl[1]
      param_dicts.append(param_dict)
    return param_dicts

  #"layers" needed for localization
  def sum(self, bottoms):
    return L.Eltwise(*bottoms, operation=1)

  def prod(self, bottoms):
    return L.Eltwise(*bottoms, operation=0)

  def rename_tops(self, tops, names):
     if not isinstance(tops, list):
       tops = [tops]
     if isinstance(names, str):
       names = [names]
     for top, name in zip(tops, names): setattr(self.n, name, top)

  def normalize(self, bottom, axis=1, numtiles=4096):
    power = L.Power(bottom, power=2)
    power_sum = L.Reduction(power, axis=axis, operation=1)
    sqrt = L.Power(power_sum, power=-0.5, shift=0.00001)
    if axis == 1:
        reshape = L.Reshape(sqrt, shape=dict(dim=[-1,1])) 
    if axis == 2:
        reshape = L.Reshape(sqrt, shape=dict(dim=[self.batch_size,-1, 1])) 
    tile = L.Tile(reshape, axis=axis, tiles=numtiles) 
    return L.Eltwise(tile, bottom, operation=0)

  #image models
  def image_model_one_layer(self, bottom, time_stamp=None, axis=1, tag=''):
    if time_stamp: 
        bottom = L.Concat(bottom, time_stamp, axis=1) #time stamp will just be zeros for --no-loc option
    inner_product = L.InnerProduct(bottom, num_output=self.visual_embedding_dim[0], 
                           weight_filler=self.uniform_weight_filler(-0.08, .08),
                           bias_filler=self.constant_filler(0), 
                           param=self.learning_params([[1,1], [2,0]], ['image_embed1'+tag, 'image_embed_1b'+tag]), 
                           axis=axis)
    dropout = L.Dropout(inner_product, dropout_ratio=self.dropout_visual)
    setattr(self.n, 'embedding_visual', dropout)
    return dropout 

  def image_model_two_layer(self, bottom, time_stamp=None, axis=1, tag=''):
    if time_stamp: 
        bottom = L.Concat(bottom, time_stamp, axis=1) #time stamp will just be zeros for --no-loc option

    inner_product_1 =  L.InnerProduct(bottom, num_output=self.visual_embedding_dim[0], 
                               weight_filler=self.uniform_weight_filler(-0.08, .08),
                               bias_filler=self.constant_filler(0), 
                               param=self.learning_params([[1,1], [2,0]], ['image_embed1'+tag, 'image_embed_1b'+tag]), axis=axis)


    if self.image_tag:
      setattr(self.n, self.image_tag + 'ip1' + str(self.count_im), inner_product_1)
      self.count_im += 1
    nonlin_1 = L.ReLU(inner_product_1)

    top_visual =  L.InnerProduct(nonlin_1, num_output=self.visual_embedding_dim[1], 
                           weight_filler=self.uniform_weight_filler(-0.08, .08),
                           bias_filler=self.constant_filler(0), 
                           param=self.learning_params([[1,1], [2,0]], ['image_embed2'+tag, 'image_embed_b2'+tag]), axis=axis)

    if self.image_tag:
      setattr(self.n, self.image_tag + 'ip2' + str(self.count_im), top_visual)
      self.count_im += 1
    dropout = L.Dropout(top_visual, dropout_ratio=self.dropout_visual)

    setattr(self.n, 'embedding_visual', dropout)
    return dropout

  #language_models
  def language_model_lstm_no_embed(self, sent_bottom, cont_bottom, text_name='embedding_text', tag=''):

    lstm_lr = self.args.lstm_lr
    embedding_lr = self.args.language_embedding_lr
      
    lstm = L.LSTM(sent_bottom, cont_bottom, 
                  recurrent_param = dict(num_output=self.language_embedding_dim[0],
                  weight_filler=self.uniform_weight_filler(-0.08, 0.08),
                  bias_filler = self.constant_filler(0)),
                  param=self.learning_params([[lstm_lr,lstm_lr], [lstm_lr,lstm_lr], [lstm_lr,lstm_lr]], ['lstm1'+tag, 'lstm2'+tag, 'lstm3'+tag])) 
    lstm_slices = L.Slice(lstm, slice_point=self.params['sentence_length']-1, axis=0, ntop=2)
    self.n.tops['silence_cell_'+str(self.silence_count)] = L.Silence(lstm_slices[0], ntop=0)
    self.silence_count += 1 
    top_lstm = L.Reshape(lstm_slices[1], shape=dict(dim=[-1, self.language_embedding_dim[0]]))
    top_text =  L.InnerProduct(top_lstm, num_output=self.language_embedding_dim[1], 
                               weight_filler=self.uniform_weight_filler(-0.08, .08),
                               bias_filler=self.constant_filler(0), 
                               param=self.learning_params([[embedding_lr,embedding_lr], [embedding_lr*2,0]], ['lstm_embed1'+tag, 'lstm_embed_1b'+tag]))

    setattr(self.n, text_name, top_text)
    return top_text

  def relational_ranking_loss(self, distance_p, distance_n, lw=1):
    """
    This function assumes you want to MINIMIZE distances
    """

    negate_distance_n = L.Power(distance_n, scale=-1)
    max_sum = L.Eltwise(distance_p, negate_distance_n, operation=1)
    max_sum_margin = L.Power(max_sum, shift=self.margin)
    max_sum_margin_relu = L.ReLU(max_sum_margin, in_place=False)
    ranking_loss = L.Reduction(max_sum_margin_relu, operation=4, loss_weight=[lw])

    return  ranking_loss

  def context_supervision_loss(self, distance, lw=1, ind_loss=None):

    """
    Distance is positive; want gt distance to be SMALLER than other distances.
    Loss used for context supervision is also ranking loss:
        Look at rank loss between all possible pairs of moments; want gt distance to be smaller.
        Take average.
    """

    slices = L.Slice(distance, ntop=21, axis=1) 
    gt = slices[0]
    setattr(self.n, 'gt_slice', gt)
    ranking_losses = []
    for i in range(1, 21):
      setattr(self.n, 'context_slice_%d' %i, slices[i])
      negate_distance = L.Power(slices[i], scale=-1)
      max_sum = L.Eltwise(gt, negate_distance, operation=1)
      max_sum_margin = L.Power(max_sum, shift=self.margin)
      max_sum_margin_relu = L.ReLU(max_sum_margin, in_place=False)
      if ind_loss:
          max_sum_margin_relu = L.Reshape(max_sum_margin_relu, shape=dict(dim=[self.batch_size, 1]))
          max_sum_margin_relu = L.Eltwise(max_sum_margin_relu, ind_loss, operation=0) 
      setattr(self.n, 'max_sum_margin_relu_%d' %i, max_sum_margin_relu)
      ranking_loss = L.Reduction(max_sum_margin_relu, operation=4)
      ranking_losses.append(ranking_loss)
    sum_ranking_losses = L.Eltwise(*ranking_losses, operation=1)
    loss = L.Power(sum_ranking_losses, scale=1/21., loss_weight=[lw])
    return loss

  def ranking_loss(self, p, n, t, lw=1):

    # I <3 Caffe - this is not obnoxious to write at all.
    distance_p = self.distance_function(p, t)
    distance_n = self.distance_function(n, t)
    negate_distance_n = L.Power(distance_n, scale=-1)
    max_sum = L.Eltwise(distance_p, negate_distance_n, operation=1)
    max_sum_margin = L.Power(max_sum, shift=self.margin)
    max_sum_margin_relu = L.ReLU(max_sum_margin, in_place=False)
    ranking_loss = L.Reduction(max_sum_margin_relu, operation=4, loss_weight=[lw])

    return  ranking_loss
 
  def write_net(self, save_file, top):
    write_proto = top.to_proto()
      
    with open(save_file, 'w') as f:
      print(write_proto, file=f)
    print("Wrote net to: %s." %save_file) 

  def get_models(self):
    if self.vision_layers == '1':
      vision_layer = self.image_model_one_layer
    elif self.vision_layers == '2':
      vision_layer = self.image_model_two_layer  
      assert len(self.visual_embedding_dim) == 2 
    else:
      raise Exception("No specified vision layer for %s" %self.vision_layers)

    if self.language_layers == 'lstm_no_embed':
      language_layer = self.language_model_lstm_no_embed
      assert len(self.language_embedding_dim) == 2 
    else:
      raise Exception("No specified language layer for %s" %self.language_layers)

    return vision_layer, language_layer

  def build_relational_model(self, param_str, save_tag):
    data = L.Python(module="data_processing", layer=self.data_layer, param_str=str(param_str), ntop=self.top_size)
    for key, value in zip(self.params['top_names_dict'].keys(), self.params['top_names_dict'].values()):
        setattr(self.n, key, data[value])
  
    im_model, lang_model = self.get_models()

    #bottoms which are always produced
    bottom_positive = data[self.top_name_dict['features_p']]
    bottom_negative = data[self.top_name_dict['features_n']]
    # 'global' is carryover name from MCN -- global == context moment here.
    global_positive = data[self.top_name_dict['features_global_p']] 

    bottom_positive_tile = L.Tile(bottom_positive, axis=1, tiles=21)          
    bottom_negative_tile = L.Tile(bottom_negative, axis=1, tiles=21)          

    concat_positive = L.Concat(bottom_positive_tile, global_positive, axis=2) 
    concat_negative = L.Concat(bottom_negative_tile, global_positive, axis=2) 

    if self.inter:
      bottom_inter = data[self.top_name_dict['features_inter']]
      global_inter = data[self.top_name_dict['features_global_inter']]
      bottom_inter_tile = L.Tile(bottom_inter, axis=1, tiles=21)          
      concat_inter = L.Concat(bottom_inter_tile, global_inter, axis=2) 

    query = data[self.top_name_dict['BoG']]
           
    bottom_positive_feature = im_model(concat_positive, axis=2)
    bottom_negative_feature = im_model(concat_negative, axis=2)
    
    if self.inter:
        bottom_inter_feature = im_model(concat_inter, axis=2)

    #'cont' is for LSTM in Caffe -- would not need this if using average Glove features.
    cont = data[self.top_name_dict['cont']]
    query = lang_model(query, cont)

    t_reshape = L.Reshape(query, shape=dict(dim=[self.batch_size, 1, -1]))
    t_tile = L.Tile(t_reshape, axis=1, tiles=21)

    #loss function
    distance_p = self.distance_function(bottom_positive_feature, t_tile) 
    distance_n = self.distance_function(bottom_negative_feature, t_tile)
    setattr(self.n, 'distance_p', distance_p[0]) 
    setattr(self.n, 'distance_n', distance_n[0]) 

    if self.inter:
        distance_inter = self.distance_function(bottom_inter_feature, t_tile)
        setattr(self.n, 'distance_inter', distance_inter[0]) 
        self.n.tops['ranking_loss_inter'] = self.relational_ranking_loss(distance_p[0], distance_inter[0], lw=self.lw_inter)
    self.n.tops['ranking_loss_intra'] = self.relational_ranking_loss(distance_p[0], distance_n[0], lw=self.lw_intra)

    if self.args.strong_supervise:
        self.n.tops['context_supervision_loss'] = self.context_supervision_loss(distance_p[1], lw=self.args.lw_strong_supervision, ind_loss=data[self.top_name_dict['strong_supervision_loss']]) 
    if self.args.stronger_supervise:
        #can also assert that the model needs to look at the correct context for the neg moment.
        self.n.tops['negative_context_supervision_loss'] = self.context_supervision_loss(distance_n[1], lw=self.args.lw_strong_supervision, ind_loss=data[self.top_name_dict['strong_supervision_loss']]) 

    self.write_net(save_tag, self.n)

  def build_relational_model_deploy(self, save_tag, visual_feature_dim, language_feature_dim):

    image_input =  L.DummyData(shape=[dict(dim=[21, 1, visual_feature_dim+2])], ntop=1) 
    setattr(self.n, 'image_data', image_input) 

    image_global =  L.DummyData(shape=[dict(dim=[21, 21, visual_feature_dim+2])], ntop=1) 
    setattr(self.n, 'global_data', image_global) 
   
    im_model, lang_model = self.get_models()

    self.silence_count += 1      

    bottom_tile = L.Tile(image_input, axis=1, tiles=21)

    bottom_concat = L.Concat(bottom_tile, image_global, axis=2)
    bottom_visual = im_model(bottom_concat, axis=2)

    text_input =  L.DummyData(shape=[dict(dim=[self.params['sentence_length'], 21, language_feature_dim])], ntop=1) 
    setattr(self.n, 'text_data', text_input)  
    cont_input =  L.DummyData(shape=[dict(dim=[self.params['sentence_length'], 21])], ntop=1) 
    setattr(self.n, 'cont_data', cont_input)  
    bottom_text = lang_model(text_input, cont_input)

    t_reshape = L.Reshape(bottom_text, shape=dict(dim=[self.batch_size, 1, -1]))
    t_tile = L.Tile(t_reshape, axis=1, tiles=21)

    self.n.tops['scores'] = self.distance_function(bottom_visual, t_tile)[0] 

    self.write_net(save_tag, self.n)

  def build_retrieval_model(self, param_str, save_tag):

    data = L.Python(module="data_processing", layer=self.data_layer, param_str=str(param_str), ntop=self.top_size)
    for key, value in zip(self.params['top_names_dict'].keys(), self.params['top_names_dict'].values()):
        setattr(self.n, key, data[value])
    
    im_model, lang_model = self.get_models()

    data_bottoms = []

    #bottoms which are always produced
    bottom_positive = data[self.top_name_dict['features_p']]
    query = data[self.top_name_dict['BoG']]
    p_time_stamp = data[self.top_name_dict['features_time_stamp_p']]
    n_time_stamp = data[self.top_name_dict['features_time_stamp_n']]
    if self.inter:
      bottom_inter = data[self.top_name_dict['features_inter']]
    if self.intra:
      bottom_intra = data[self.top_name_dict['features_intra']]

    bottom_positive = im_model(bottom_positive, p_time_stamp)
    if self.inter:
      bottom_inter = im_model(bottom_inter, p_time_stamp)
    if self.intra:
      bottom_intra = im_model(bottom_intra, n_time_stamp)
    if (self.inter) & (not self.intra):
      self.n.tops['silence_cell_'+str(self.silence_count)] = L.Silence(n_time_stamp, ntop=0)
      self.silence_count += 1      

    cont = data[self.top_name_dict['cont']]
    query = lang_model(query, cont)

    if not args.tall_loss:
        if self.inter:
          self.n.tops['ranking_loss_inter'] = self.ranking_loss(bottom_positive, bottom_inter, query, lw=self.lw_inter)
        if self.intra:
          self.n.tops['ranking_loss_intra'] = self.ranking_loss(bottom_positive, bottom_intra, query, lw=self.lw_intra)
    else:
        if self.inter:
          self.n.tops['tall_loss_inter'] = self.tall_loss(bottom_positive, bottom_inter, query, lw=self.lw_inter)
        if self.intra:
          self.n.tops['tall_loss_intra'] = self.tall_loss(bottom_positive, bottom_intra, query, lw=self.lw_intra)

    self.write_net(save_tag, self.n)

  def build_retrieval_model_deploy(self, save_tag, visual_feature_dim, language_feature_dim):

    image_input =  L.DummyData(shape=[dict(dim=[21, visual_feature_dim])], ntop=1) 
    setattr(self.n, 'image_data', image_input) 

    loc_input =  L.DummyData(shape=[dict(dim=[21, 2])], ntop=1) 
    setattr(self.n, 'loc_data', loc_input) 
   
    im_model, lang_model = self.get_models()

    bottom_visual = im_model(image_input, loc_input)

    if self.language_layers in recurrent_layers:

      text_input =  L.DummyData(shape=[dict(dim=[self.params['sentence_length'], 21, language_feature_dim])], ntop=1) 
      setattr(self.n, 'text_data', text_input)  
      cont_input =  L.DummyData(shape=[dict(dim=[self.params['sentence_length'], 21])], ntop=1) 
      setattr(self.n, 'cont_data', cont_input)  
      bottom_text = lang_model(text_input, cont_input)

    else:
      text_input =  L.DummyData(shape=[dict(dim=[21, language_feature_dim])], ntop=1) 
      bottom_text = lang_model(text_input)
      if self.language_layers == '0': 
        setattr(self.n, 'text_data', bottom_text)  
      else:
        setattr(self.n, 'text_data', text_input)  

    self.n.tops['rank_score'] = self.distance_function(bottom_visual, bottom_text)
    self.write_net(save_tag, self.n)

def make_solver(save_name, snapshot_prefix, train_nets, test_nets, **kwargs):

  #set default values
  parameter_dict = kwargs
  if 'test_iter' not in parameter_dict.keys(): parameter_dict['test_iter'] = 10
  if 'test_interval' not in parameter_dict.keys(): parameter_dict['test_interval'] = 100
  if 'base_lr' not in parameter_dict.keys(): parameter_dict['base_lr'] = 0.1
  if 'lr_policy' not in parameter_dict.keys(): parameter_dict['lr_policy'] = '"step"' 
  if 'display' not in parameter_dict.keys(): parameter_dict['display'] = 100 
  if 'max_iter' not in parameter_dict.keys(): parameter_dict['max_iter'] = 10000
  if 'gamma' not in parameter_dict.keys(): parameter_dict['gamma'] = 0.1
  if 'stepsize' not in parameter_dict.keys(): parameter_dict['stepsize'] = 5000
  if 'snapshot' not in parameter_dict.keys(): parameter_dict['snapshot'] = 2500
  if 'momentum' not in parameter_dict.keys(): parameter_dict['momentum'] = 0.9
  if 'weight_decay' not in parameter_dict.keys(): parameter_dict['weight_decay'] = 0.0
  if 'solver_mode' not in parameter_dict.keys(): parameter_dict['solver_mode'] = 'GPU'
  if 'random_seed' not in parameter_dict.keys(): parameter_dict['random_seed'] = 1701
  if 'average_loss' not in parameter_dict.keys(): parameter_dict['average_loss'] = 100
  if 'clip_gradients' not in parameter_dict.keys(): parameter_dict['clip_gradients'] = 10
  if 'device_id' not in parameter_dict.keys(): parameter_dict['device_id'] = 0 
  if 'debug_info' not in parameter_dict.keys(): parameter_dict['debug_info'] = 'false'

  if parameter_dict['type'] == '"Adam"':
      parameter_dict['lr_policy'] = '"fixed"' 
      parameter_dict['momentum2'] = 0.999
      parameter_dict['regularization_type'] = '"L2"'
      if 'type' not in parameter_dict.keys(): parameter_dict['delta'] = 0.0000001 

  snapshot_prefix = 'snapshots/%s' %snapshot_prefix
  parameter_dict['snapshot_prefix'] = '"%s"' %snapshot_prefix
 
  write_txt = open(save_name, 'w')
  write_txt.writelines('train_net: "%s"\n' %train_nets)
  for tn in test_nets:
    write_txt.writelines('test_net: "%s"\n' %tn)
    write_txt.writelines('test_iter: %d\n' %parameter_dict['test_iter'])
  if len(test_nets) > 0:
    write_txt.writelines('test_interval: %d\n' %parameter_dict['test_interval'])

  parameter_dict.pop('test_iter')
  parameter_dict.pop('test_interval')

  for key in parameter_dict.keys():
    write_txt.writelines('%s: %s\n' %(key, parameter_dict[key]))
  write_txt.close()
  print("Wrote solver to %s." %save_name)

def train_model(solver_path, net=None):
  solver = caffe.get_solver(solver_path)
  if net:
    solver.net.copy_from(net)
    print("Copying weights from %s" %net)
  solver.solve()
 
if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  #So many parameters to define your net!

  #how to tag built nets/snapshots etc.
  parser.add_argument("--tag", type=str, default='') 
  parser.add_argument("--snapshot_folder", type=str, default='snapshots') 
  parser.add_argument('--train', dest='train', action='store_true') #train net or test current network
  parser.set_defaults(train=False)

  #training data
  parser.add_argument("--train_json", type=str, default='data/prep_data_train.json') 
  parser.add_argument("--train_h5", type=str, default='data/average_fc7_feats_fps.h5') 
  parser.add_argument("--test_json", type=str, default='data/prep_data_val_all.json') 
  parser.add_argument("--test_h5", type=str, default='data/average_fc7_feats_fps.h5') 

  #net specifications
  parser.add_argument("--feature_process_visual", type=str, default='feature_process_norm') 
  parser.add_argument("--feature_process_language", type=str, default='average_glove') 
  parser.add_argument('--loc', dest='loc', action='store_true')
  parser.add_argument('--no-loc', dest='loc', action='store_false')
  parser.set_defaults(loc=False)
  parser.add_argument('--loss_type', type=str, default='triplet')
  parser.add_argument('--margin', type=float, default=0.1)
  parser.add_argument('--input_visual_data', type=str, default='clip')
  parser.add_argument('--dropout_visual', type=float, default=0.0)
  parser.add_argument('--dropout_language', type=float, default=0.0)
  parser.add_argument('--visual_embedding_dim', type=int, nargs='+', default=[100, 100])
  parser.add_argument('--language_embedding_dim', type=int, nargs='+', default=[100])
  parser.add_argument('--lw_inter', type=float, default=None)
  parser.add_argument('--lw_intra', type=float, default=None)

  parser.add_argument('--lw_strong_supervision', type=float, default=1.)
  parser.add_argument('--global_supervision', dest='global_supervision', action='store_true')
  parser.set_defaults(global_supervision=False)
  parser.add_argument('--decay_context_supervision', dest='decay_context_supervision', action='store_true')
  parser.set_defaults(decay_context_supervision=False)
  parser.add_argument('--vision_layers', type=str, default='1')
  parser.add_argument('--language_layers', type=str, default='1')
  parser.add_argument('--distance_function', type=str, default='euclidean_distance')
  parser.add_argument('--no_strong_supervise', dest='strong_supervise', action='store_false') 
  parser.add_argument('--strong_supervise', dest='strong_supervise', action='store_true') 
  parser.set_defaults(strong_supervise=False)
  parser.add_argument('--no_context_tef', dest='context_tef', action='store_false') 
  parser.add_argument('--context_tef', dest='context_tef', action='store_true') 
  parser.set_defaults(context_tef=False)
  parser.add_argument('--context_supervise', dest='strong_supervise', action='store_false') 
  parser.add_argument('--stronger_supervise', dest='stronger_supervise', action='store_true') 
  parser.set_defaults(stronger_supervise=False)
  parser.add_argument('--strong_supervise_test', dest='strong_supervise_test', action='store_true') 
  parser.set_defaults(strong_supervise_test=False)
  parser.add_argument('--pool_type', type=str, default='max')
  parser.add_argument('--tall_loss', dest='tall_loss', action='store_true') 
  parser.set_defaults(tall_loss=False)
  parser.add_argument('--tall_distance_size', type=int, default=1) 
  parser.add_argument('--concat_tef', dest='concat_tef', action='store_true') 
  parser.set_defaults(concat_tef=False)

  #solver specifications
  parser.add_argument('--random_seed', type=int, default='1701')
  parser.add_argument('--python_random_seed', type=int, default=10)
  parser.add_argument('--max_iter', type=int, default=10000)
  parser.add_argument('--snapshot', type=int, default=5000)
  parser.add_argument('--stepsize', type=int, default=5000)
  parser.add_argument('--base_lr', type=float, default=0.01)
  parser.add_argument('--lstm_lr', type=float, default=10)
  parser.add_argument('--language_embedding_lr', type=float, default=1)
  parser.add_argument('--batch_size', type=int, default=120)
  parser.add_argument('--weight_decay', type=float, default=0)
  parser.add_argument('--pretrained_model', type=str, default=None)
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--image_tag', type=str, default=None)
  parser.add_argument('--solver_type', type=str, default='"SGD"')
  parser.add_argument('--delta', type=float, default=1e-8)


  args = parser.parse_args()

  print("Feature process visual: %s" %args.feature_process_visual)
  print("Feature process language: %s" %args.feature_process_language)
  print("Loc: %s" %args.loc)
  print("Dropout visual %f" %args.dropout_visual)
  print("Dropout language %f" %args.dropout_language)
  print("Pretrained model %s" %args.pretrained_model)
  valid_visual_process = ['feature_process_base', 'feature_process_mean_subtraction', 
                           'feature_process_norm', 'zero_feature',
                           'feature_process_context', 'feature_process_lstm',
                           'feature_process_norm_combine', 
                           'feature_process_before_after',
                           'feature_process_before']
  valid_language_process = ['BoW', 'average_glove', 'recurrent_word', 'zero_language', 'recurrent_embedding']
  valid_loss_type = ['triplet', 'inter', 'intra']
  valid_input_visual_data = ['clip', 'video', 'relational']
  
  assert args.feature_process_visual in valid_visual_process
  assert args.feature_process_language in valid_language_process
  assert args.loss_type in valid_loss_type
  assert args.input_visual_data in valid_input_visual_data

  #define loss weights
  if args.loss_type == 'inter':
    args.lw_inter = 1
    args.lw_intra = 0 

  elif args.loss_type == 'intra':
    args.lw_intra = 1
    args.lw_inter = 0 

  else: #triplet
    if not args.lw_intra:
      args.lw_intra = 1 - args.lw_inter
    if not args.lw_inter:
      args.lw_inter = 1 - args.lw_intra

  assert args.lw_inter >= 0
  assert args.lw_intra >= 0

  train_base = 'prototxts/%s_train.prototxt'
  solver_base = 'prototxts/%s_solver.prototxt' 
  deploy_base = 'prototxts/%s_deploy.prototxt' 
  snapshot_base = '' 
  
  params = {}
  params['sentence_length'] = 50
  params['descriptions'] = args.train_json 
  params['features'] = args.train_h5 
  params['feature_process'] = args.feature_process_visual
  params['loc_feature'] = args.loc 
  params['language_feature'] = args.feature_process_language 
  params['loss_type'] = args.loss_type
  params['batch_size'] = args.batch_size  
  params['inputVisualData'] = args.input_visual_data
  params['strong_supervision'] = args.strong_supervise
  params['strong_supervision_test'] = args.strong_supervise_test
  params['global_supervision'] = args.global_supervision
  params['decay_context_supervision'] = args.decay_context_supervision
  params['python_random_seed'] = args.python_random_seed
  params['context_tef'] = args.context_tef

  #Create a dict with all the top names for the data layer.
  if args.input_visual_data == 'clip':
    params['top_names'] = ['features_p', 'BoG', 'features_time_stamp_p', 'features_time_stamp_n']

    if args.loss_type in ['triplet', 'inter']:
      inter_top_name = 'features_inter'
      params['top_names'].append(inter_top_name)
    if args.loss_type in ['triplet', 'intra']:
      intra_top_name = 'features_intra'
      params['top_names'].append(intra_top_name)
  elif args.input_visual_data == 'relational':
    params['top_names'] = ['features_p', 'features_n', 'features_global_p', 'BoG']
    if args.lw_inter > 0:
      params['top_names'].extend(['features_inter', 'features_global_inter'])
  else:
    raise Exception("Did not select valid input visual data type.")

  params['top_names_dict'] = {}    
  for top_name in params['top_names']: 
    params['top_names_dict'] = add_dict_values(top_name, params['top_names_dict'])

  if args.language_layers in recurrent_layers:
    params['top_names'].append('cont')
    params['top_names_dict'] = add_dict_values('cont', params['top_names_dict'])
    params['sentence_length'] = 50
    assert params['language_feature'] in ['recurrent_word', 'recurrent_embedding'] 

  if args.strong_supervise or args.stronger_supervise:
    params['top_names'].append('strong_supervision_loss')
    params['top_names_dict'] = add_dict_values('strong_supervision_loss', params['top_names_dict'])

  top_size = len(params['top_names'])

  #get size of visual features 
  f = h5py.File(params['features'])
  feat = np.array(f.values()[0]) 
  f.close()
  visual_feature_dim = feature_process_dict[args.feature_process_visual](0,0,feat).shape[-1]

  #get size of vocab
  language_processor = language_feature_process_dict[params['language_feature']](read_json(params['descriptions'])) 
  language_feature_dim = language_processor.get_vector_dim() 
  vocab_size = language_processor.get_vocab_size() 
  params['vocab_size'] = vocab_size
 
  pretrained_model_bool = False
  if args.pretrained_model:
    pretrained_model_bool = True 

  train_path = train_base %args.tag
  deploy_path = deploy_base %args.tag 
  solver_path = solver_base %args.tag 
  
  net = retrieval_net(args=args, param_str=params,params=params, top_size=top_size)
  net.visual_feature_dim = visual_feature_dim
  if args.input_visual_data == 'relational':
    net.build_relational_model(params, train_path) 
  else:
    net.build_retrieval_model(params, train_path) 
  
  params['batch_size'] = 100

  net = retrieval_net(args=args, param_str=params,params=params, top_size=top_size, is_test=True)
  net.visual_feature_dim = visual_feature_dim
  net.batch_size=21
  if args.input_visual_data == 'relational':
    net.build_relational_model_deploy(deploy_path, visual_feature_dim, language_feature_dim) 
  else:
    net.build_retrieval_model_deploy(deploy_path, visual_feature_dim, language_feature_dim) 

  max_iter = args.max_iter 
  snapshot = args.snapshot
  stepsize = args.stepsize
  base_lr = args.base_lr 
  if args.train and os.path.exists(solver_path):
    #Let's not overwrite our old snapshots if we are training a new model.
    print("Already have a solver path with this name (%s); continue if you would like to overwrite prototxts and snapshots with this name." %solver_path)
    import pdb; pdb.set_trace()
   
  make_solver(solver_path, args.tag, train_path, [], **{'device_id': args.gpu, 'max_iter': max_iter, 'snapshot': snapshot, 'weight_decay': args.weight_decay, 'stepsize': stepsize, 'base_lr': base_lr, 'random_seed': args.random_seed, 'display': 10, 'type': args.solver_type, 'delta': args.delta, 'iter_size': 120/args.batch_size})
  caffe.set_device(args.gpu)
  caffe.set_mode_gpu()
  caffe.set_device(args.gpu)
 
  if args.train:
    train_model(solver_path, args.pretrained_model)
  elif args.input_visual_data == 'relational':
    test_model(deploy_path, args.tag, max_iter=max_iter, snapshot_interval=snapshot, test_h5=args.test_h5,test_json=args.test_json, params=params, scores_layer='scores', save_context=True, snapshot_folder=args.snapshot_folder) 
  else:
    test_model(deploy_path, args.tag, max_iter=max_iter, snapshot_interval=snapshot, test_h5=args.test_h5,test_json=args.test_json, params=params, snapshot_folder=args.snapshot_folder)
