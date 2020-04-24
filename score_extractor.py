import sys
import os
import json
import numpy as np
import pdb

parent = './dumped/test_xnli_mlm_tlm_contrastive'

folders = [x[0] for x in os.walk(parent) if x[0] != parent]
scores = {}
scores['best'] = None

for folder in folders:
    directory = folder + '/train.log'
    f = open(directory, 'r')
    params = []
    for line in f:
        # pdb.set_trace()
        if 'optimizer_e:' in line and len(params) <= 4:
            params.append(line.strip().split('optimizer_e:')[1].strip())
        if 'optimizer_p:' in line and len(params) <= 4:
            params.append(line.strip().split('optimizer_p:')[1].strip())
        if ' batch_size:' in line and len(params) <= 4:
            params.append(line.strip().split('batch_size:')[1].strip())
        if 'epoch_size:' in line and len(params) <= 4:
            params.append(line.strip().split('epoch_size:')[1].strip())
        if len(params) == 4 and (str(params) not in scores.keys()):
            print(params) 
            scores[str(params)] = {}
        if '__log__' in line:
            score_dict = line.split('__')[-1][1:]
            score_dict = json.loads(score_dict)
            epoch = score_dict['epoch']
            del score_dict['epoch']
            scores[str(params)][epoch] = score_dict
            val_score = np.mean([v for k,v in score_dict.items() if 'valid' in k])
            scores[str(params)][epoch]['avg_valid'] = val_score
            if scores['best'] is None or scores['best']['avg_valid'] < val_score:
                if scores['best'] is None:
                    scores['best'] = {}
                scores['best']['optim'] = str(params)
                scores['best']['avg_valid'] = val_score
                scores['best']['epoch'] = epoch
                scores['best']['dict'] = score_dict


best_scores = {}
for k in scores.keys():
    if k != 'best':
        print("OPTIMIZER: {}".format(str(k)))
        max_val = 0
        max_dict = None
        for epoch in scores[k]:
            if scores[k][epoch]['avg_valid'] > max_val:
                max_val = scores[k][epoch]['avg_valid']
                max_dict = scores[k][epoch]
        best_scores[k] = max_dict
        print(max_dict)
print('==================================================')
print("BEST SCORES: {}".format(scores['best']))


with open(parent + '/results_best_per_optim.txt', 'w') as json_file:
    json.dump(best_scores, json_file)
with open(parent + '/results.txt', 'w') as json_file:
    json.dump(scores, json_file)
with open(parent + '/results_best.txt', 'w') as json_file:
    json.dump(scores['best'], json_file)