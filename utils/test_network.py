import sys
import os
sys.path.append('caffe/python/')
import caffe
import h5py
import itertools
import numpy as np
import random
import copy
from data_processing import *
from python_utils import *

results_file = 'cache_results/'
if not os.path.exists(results_file):
   os.mkdir(results_file)

feature_process_dict = {'feature_process_base': feature_process_base,
                        'feature_process_norm': feature_process_norm,
                        'feature_process_context': feature_process_context,
                        'feature_process_before_after': feature_process_before_after
                        }

language_feature_process_dict = {'recurrent_embedding': recurrent_embedding}

def iou(pred, gt):
  intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
  union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
  return float(intersection)/union

def rank(pred, gt):
  try:
      return pred.index(tuple(gt)) + 1 
  except:
      return 21 

def eval_predictions(segments, data, quiet=False):
  average_ranks = []
  average_iou = []
  for s, d in zip(segments, data):
    pred = s[0]
    ious = [iou(pred, t) for t in d['times']]
    average_iou.append(np.mean(np.sort(ious)[-3:])) 
    ranks = [rank(s, t) for t in d['times']]
    average_ranks.append(np.mean(np.sort(ranks)[:3]))
  rank1 = np.sum(np.array(average_ranks) <= 1)/float(len(average_ranks)) 
  rank3 = np.sum(np.array(average_ranks) <= 3)/float(len(average_ranks))
  rank5 = np.sum(np.array(average_ranks) <= 5)/float(len(average_ranks))
  miou = np.mean(average_iou)

  if not quiet:
      print "Average rank@1: %f" %rank1
      print "Average rank@3: %f" %rank3
      print "Average rank@5: %f" %rank5
      print "Average iou: %f" %miou
  return rank1, rank5, miou

def write_raw_results(iter, data, sorted_segments_list, test_json, snapshot_tag):
  if 'test' in test_json:
    test_or_train = 'test'
  elif 'val' in test_json:
    test_or_train = 'val'
  else:
    test_or_train = 'train'
  write_txt_file = '%s/%s_%s_raw_results_iter%s.txt' %(results_file, 
                                                       snapshot_tag, 
                                                       test_or_train, 
                                                       iter)
  write_txt = open(write_txt_file, 'w')
  for d, s in zip(data, sorted_segments_list): 
    video = d['video'] 
    annotation_id = d['annotation_id']
    write_txt.writelines('%s\t%s\t%s\n' %(video, annotation_id, s))
  print "Wrote raw results to: %s" %write_txt_file
  write_txt.close()

def write_raw_scores(iter, data, sorted_segments_list, test_json, snapshot_tag):
  if 'test' in test_json:
    test_or_train = 'test'
  elif 'val' in test_json:
    test_or_train = 'val'
  else:
    test_or_train = 'train'
  write_txt_file = '%s/%s_%s_raw_scores_iter%s.txt' %(results_file,
                                                      snapshot_tag, 
                                                      test_or_train, 
                                                      iter)
  write_txt = open(write_txt_file, 'w')
  for d, s in zip(data, sorted_segments_list): 
    video = d['video'] 
    annotation_id = d['annotation_id']
    write_txt.writelines('%s\t%s\t%s\n' %(video, annotation_id, str(list(s))))
  print "Wrote raw results to: %s" %write_txt_file
  write_txt.close()


def test_model(deploy_net, snapshot_tag, 
               max_iter=10000, 
               snapshot_interval=5000, 
               loc=True,  
               test_h5='data/average_fc7_feats_fps.h5', 
               test_json='data/prep_data_val_all.json', 
               params={},
               scores_layer='rank_score',
               save_context=False,
               snapshot_folder='snapshots'):

    #prep features to create vision/language feature extractors 
    language_extractor_fcn = extractRecurrentLanguageFeaturesEfficient
    params['bog_key'] = 'BoG'
    params['cont_key'] = 'cont'
  
    if params['inputVisualData'] == 'clip':
        visual_extractor_fcn = extractAverageClipFeatures  
    else:
        visual_extractor_fcn = extractRelationalClipFeatures 
  
    language_process = language_feature_process_dict[params['language_feature']] 
    data_orig = read_json(test_json)
    language_processor = language_process(data_orig)
    data = language_processor.preprocess(data_orig)
    params['vocab_dict'] = language_processor.vocab_dict
    num_glove_centroids = language_processor.get_vector_dim()
    params['num_glove_centroids'] = num_glove_centroids
    thread_result = {}
  
    visual_feature_extractor = visual_extractor_fcn(data, params, thread_result)
    textual_feature_extractor = language_extractor_fcn(data, params, thread_result)
    possible_segments = visual_feature_extractor.possible_annotations
  
    snapshot = '%s/%s_iter_%%d.caffemodel' %(snapshot_folder, snapshot_tag)
  
    all_sorted_segments = []
    all_scores = []
    if save_context:
        assert params['inputVisualData'] == 'relational'
        all_context_segments = []
        all_context_raw_values = []

    for iter in range(snapshot_interval, max_iter+1, snapshot_interval):
        net = caffe.Net(deploy_net, snapshot %iter, caffe.TEST)  

        sorted_segments_list = []
        raw_scores = []
        if save_context:
            context_raw_values_list = []
            context_segments_list = []

        for id, d in enumerate(data):
            sys.stdout.write("\r%d/%d" %(id, len(data)))
  
            #get data
            vis_features = visual_feature_extractor.get_data_test(d)
            lang_features, cont = textual_feature_extractor.get_data_test(d)
    
            if params['inputVisualData'] == 'relational':
                net.blobs['image_data'].data[:,0,:] = vis_features.copy()        
                for i in range(vis_features.shape[0]):

                  if params['strong_supervision_test']:
                      if d['context']:
                          idx = visual_feature_extractor.possible_annotations_dict[tuple(d['context'])]
                          net.blobs['global_data'].data[i,:,:] = np.tile(vis_features[idx,:].copy(), (21,1))        
                      else:
                          net.blobs['global_data'].data[i,:,:] = vis_features.copy()
                  else: 
                     net.blobs['global_data'].data[i,:,:] = vis_features.copy()
            else:
                net.blobs['image_data'].data[:,:] = vis_features[0].copy()        
                net.blobs['loc_data'].data[:,:] = vis_features[1].copy()        

            for i in range(len(possible_segments)):
                net.blobs['text_data'].data[:,i,:] = lang_features
                net.blobs['cont_data'].data[:,i] = cont 
    
            #run net forward
            scores = net.forward()

            #get sorted segments
            sorted_segments = [possible_segments[i] for i in np.argsort(net.blobs[scores_layer].data.squeeze())]

            #used to save context for some evaluations
            if save_context:
                #hard coded -- would be nice to have pretty name                
                context_layer = 'Reshape4' 
                context_segments = [[possible_segments[ii] for ii in np.argsort(net.blobs[context_layer].data[i,:].squeeze())] for i in np.argsort(net.blobs[scores_layer].data.squeeze())]
                context_raw_values = net.blobs[context_layer].data.squeeze().copy()
                context_segments_list.append(context_segments) 
                context_raw_values_list.append(context_raw_values) 

            sorted_segments_list.append(sorted_segments) 
            raw_scores.append(net.blobs[scores_layer].data.copy().squeeze())
        if save_context:
            all_context_segments.append(context_segments_list) 
            all_context_raw_values.append(context_raw_values_list) 
        all_sorted_segments.append(sorted_segments_list) 
        all_scores.append(raw_scores)  
 
    ###Print eval and save eval to txt files
    count = 0
    for iter in range(snapshot_interval, max_iter+1, snapshot_interval):
      print "-----------------------Iteration: %d--------------------" %iter
      rank1, rank5, miou = eval_predictions(all_sorted_segments[count], data)
      write_raw_results(iter, data, all_sorted_segments[count], test_json, snapshot_tag)
      if save_context:
          write_raw_results(iter, data, all_context_segments[count], test_json, 'context_'+snapshot_tag)
          context_dict = {}
          if 'test' in test_json:
            test_or_train = 'test'
          elif 'val' in test_json:
            test_or_train = 'val'
          else:
            test_or_train = 'train'
          for t, v in zip(data, all_context_raw_values[count]):
              context_dict[t['annotation_id']] = v
          pkl.dump(context_dict, open('context_pkls/%s_%s_%d.p' %(snapshot_tag, test_or_train, iter), 'wb')) 
      write_raw_scores(iter, data, all_scores[count], test_json, snapshot_tag)
      count += 1
