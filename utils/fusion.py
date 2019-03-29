import sys
sys.path.append('utils/')
from python_utils import *
from test_network import *
import numpy as np
import argparse
  
parser = argparse.ArgumentParser()
  
parser.add_argument("--rgb_tag", type=str, default='') 
parser.add_argument("--flow_tag", type=str, default='') 
parser.add_argument("--dataset", type=str, default='') 
parser.add_argument("--iter", type=int, default=45000) 
parser.add_argument('--tall', dest='tall', action='store_true') #train net or test current network
parser.set_defaults(tall=False)
parser.add_argument("--set", type=str, default='val+test')
parser.add_argument('--quiet', dest='quiet', action='store_true') #train net or test current network
parser.set_defaults(quiet=False)
  
args = parser.parse_args()

if args.set not in ['val+test', 'test', 'val']:
    raise Exception ("Did not select valid set (must be 'val+test', 'val', or 'test')")
if args.dataset not in ['tempoTL', 'tempoHL', 'didemo']:
    raise Exception ("Did not select valid set (must be 'val+test', 'val', or 'test')")

if args.dataset in ['tempoTL', 'tempoHL']:
    args.dataset += '+didemo'

def read_results_file(results_file):

    results_txt = open(results_file).readlines()
    results_txt = [r.strip() for r in results_txt]
    results = [r.split('\t') for r in results_txt]
    result_dict = {}
    for r in results:
        annotation_id = r[1]
        segments = eval(r[2])
        result_dict[annotation_id] = segments

    return result_dict

def get_data(split):

    data = json.load(open('data/%s_%s.json' %(args.dataset, split), 'r'))
    if 'tempoTL' in args.dataset:

        eval_datasets = [(('didemo'), [d for d in data if d['annotation_id'].split('_')[0] not in \
                         set([u'before', u'after', u'then'])]),
                         (('before'), [d for d in data if u'before' in d['annotation_id']]),
                         (('after'), [d for d in data if u'after' in d['annotation_id']]),
                         (('then'), [d for d in data if u'then' in d['annotation_id']])]
    elif 'tempoHL' in args.dataset:

        eval_datasets = [(('didemo'), [d for d in data if d['annotation_id'].split('_')[0] not in \
                         set([u'before', u'after', u'then', u'while'])]),
                         (('before'), [d for d in data if u'before' in d['annotation_id']]),
                         (('after'), [d for d in data if u'after' in d['annotation_id']]),
                         (('then'), [d for d in data if u'then' in d['annotation_id']]),
                         (('while'), [d for d in data if u'while' in d['annotation_id']])]
    else:
        eval_datasets = [(('didemo'), data)]
    return eval_datasets

split = 'val'
eval_datasets = get_data(split)
rgb_results_file = 'cache_results/%s_%s_raw_scores_iter%d.txt' %(args.rgb_tag, split, args.iter)
rgb_result_dict = read_results_file(rgb_results_file)

flow_results_file = 'cache_results/%s_%s_raw_scores_iter%d.txt' %(args.flow_tag, split, args.iter)
flow_result_dict = read_results_file(flow_results_file)

possible_segments = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5)]
for i in itertools.combinations(range(6), 2):
    possible_segments.append(i)

#val split
lams = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
track_rank1 = []

for lam in lams:
    ranks_1 = []
    ranks_5 = []
    mious = []
    
    if not args.quiet:
        print "Lambda is %f" %lam

    result_dict = {}
    for key in flow_result_dict.keys():
        scores = lam * np.array(rgb_result_dict[str(key)]) + (1-lam) * np.array(flow_result_dict[str(key)])
        result_dict[key] = [possible_segments[i] for i in np.argsort(scores)]
       
    for dataset in eval_datasets:
        val_data = dataset[1] 
        segments = []
        for d in val_data:
            annotation_id = str(d['annotation_id'])
            if args.tall:
                segments.append(result_dict[annotation_id][::-1])
            else:
                segments.append(result_dict[annotation_id])
        rank1, rank5, miou = eval_predictions(segments, val_data, quiet=True)
        ranks_1.append(rank1)
        ranks_5.append(rank5)
        mious.append(miou)
    if not args.quiet:
        print "Average Rank@1: %f" %np.mean(ranks_1) 
        print "Average Rank@5: %f" %np.mean(ranks_5) 
        print "Average mIoU: %f" %np.mean(mious) 
    track_rank1.append(np.mean(ranks_1))

lam = lams[np.argmax(track_rank1)]

if not args.quiet:
    print "BEST LAMBDA IS: %f" %lam

split = 'test'
if args.set == 'val':
    split = 'val' #only report results on val for didemo
eval_datasets = get_data(split)

rgb_results_file = 'cache_results/%s_%s_raw_scores_iter%d.txt' %(args.rgb_tag, split, args.iter)
rgb_result_dict = read_results_file(rgb_results_file)

flow_results_file = 'cache_results/%s_%s_raw_scores_iter%d.txt' %(args.flow_tag, split, args.iter)
flow_result_dict = read_results_file(flow_results_file)

result_dict = {}
for key in rgb_result_dict.keys():
    scores = lam * np.array(rgb_result_dict[key]) + (1-lam) * np.array(flow_result_dict[key])
    result_dict[key] = [possible_segments[i] for i in np.argsort(scores)]
   
for dataset in eval_datasets:
    print "##########EVALUATING FOR DATASET: %s (%s; lambda = %0.02f) #####################" %(dataset[0], split, lam)
    val_data = dataset[1] 
    segments = []
    annotation_id_to_data = {} 
    for d in val_data: annotation_id_to_data[d['annotation_id']] = d
    for d in val_data:
        annotation_id = str(d['annotation_id'])
        if args.tall:
            segments.append(result_dict[annotation_id][::-1])
        else:
            segments.append(result_dict[annotation_id])
    rank1, rank5, miou = eval_predictions(segments, val_data, quiet=False)
    print "%0.05f\t%0.05f\t%0.05f" %(rank1, rank5, miou)
