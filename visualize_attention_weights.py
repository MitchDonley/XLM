import os
import torch
import numpy as np
from src.evaluation.xnli import XNLI
from src.model.embedder import SentenceEmbedder
import argparse
import pdb
from matplotlib import pyplot as plt
import seaborn as sn
from src.utils import bool_flag, initialize_exp
from src.data.dataset import ParallelDataset
from src.data.loader import load_binarized, set_dico_parameters

XNLI_LANGS = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']

def load_data(parameters):
    """
    Load XNLI cross-lingual classification data.
    """
    params = parameters
    data = {lang: {splt: {} for splt in ['train', 'valid', 'test']} for lang in XNLI_LANGS}
    label2id = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    dpath = os.path.join(params.data_path, 'eval', 'XNLI')

    for splt in ['train', 'valid', 'test']:

        for lang in XNLI_LANGS:

            # only English has a training set
            if splt == 'train' and lang != 'en':
                del data[lang]['train']
                continue

            # load data and dictionary
            data1 = load_binarized(os.path.join(dpath, '%s.s1.%s.pth' % (splt, lang)), params)
            data2 = load_binarized(os.path.join(dpath, '%s.s2.%s.pth' % (splt, lang)), params)
            data['dico'] = data.get('dico', data1['dico'])

            # set dictionary parameters
            set_dico_parameters(params, data, data1['dico'])
            set_dico_parameters(params, data, data2['dico'])

            # create dataset
            data[lang][splt]['x'] = ParallelDataset(
                data1['sentences'], data1['positions'],
                data2['sentences'], data2['positions'],
                params
            )

            # load labels
            with open(os.path.join(dpath, '%s.label.%s' % (splt, lang)), 'r') as f:
                labels = [label2id[l.rstrip()] for l in f]
            data[lang][splt]['y'] = torch.LongTensor(labels)
            assert len(data[lang][splt]['x']) == len(data[lang][splt]['y'])

    return data

# parse parameters
parser = argparse.ArgumentParser(description='Train on GLUE or XNLI')

# main parameters
parser.add_argument("--exp_name", type=str, default="",
                    help="Experiment name")
parser.add_argument("--dump_path", type=str, default="",
                    help="Experiment dump path")
parser.add_argument("--exp_id", type=str, default="",
                    help="Experiment ID")

# evaluation task / pretrained model
parser.add_argument("--transfer_tasks", type=str, default="",
                    help="Transfer tasks, example: 'MNLI-m,RTE,XNLI' ")
parser.add_argument("--model_path", type=str, default="",
                    help="Model location")

# data
parser.add_argument("--data_path", type=str, default="",
                    help="Data path")
parser.add_argument("--max_vocab", type=int, default=-1,
                    help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--min_count", type=int, default=0,
                    help="Minimum vocabulary count")

# batch parameters
parser.add_argument("--max_len", type=int, default=256,
                    help="Maximum length of sentences (after BPE)")
parser.add_argument("--group_by_size", type=bool_flag, default=False,
                    help="Sort sentences by size during the training")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Number of sentences per batch")
parser.add_argument("--max_batch_size", type=int, default=0,
                    help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
parser.add_argument("--tokens_per_batch", type=int, default=-1,
                    help="Number of tokens per batch")

# model / optimization
parser.add_argument("--finetune_layers", type=str, default='0:_1',
                    help="Layers to finetune. 0 = embeddings, _1 = last encoder layer")
parser.add_argument("--weighted_training", type=bool_flag, default=False,
                    help="Use a weighted loss during training")
parser.add_argument("--dropout", type=float, default=0,
                    help="Fine-tuning dropout")
parser.add_argument("--optimizer_e", type=str, default="adam,lr=0.0001",
                    help="Embedder (pretrained model) optimizer")
parser.add_argument("--optimizer_p", type=str, default="adam,lr=0.0001",
                    help="Projection (classifier) optimizer")
parser.add_argument("--n_epochs", type=int, default=100,
                    help="Maximum number of epochs")
parser.add_argument("--epoch_size", type=int, default=-1,
                    help="Epoch size (-1 for full pass over the dataset)")
parser.add_argument("--contrastive_type", type=str, default="first",
                        help="Type of sentence embeddings during contrastive learning")
parser.add_argument("--lambda_mult", type=float, default=1,
                        help="Multiplier for the nt-xent loss")
parser.add_argument("--embedding_type", type=str, default="first",
                        help="Type of sentence embeddings for down stream task")

# debug
parser.add_argument("--debug_train", type=bool_flag, default=False,
                    help="Use valid sets for train sets (faster loading)")
parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                    help="Debug multi-GPU / multi-node within a SLURM job")

# parse parameters
params = parser.parse_args()
if params.tokens_per_batch > -1:
    params.group_by_size = True

# reload pretrained model
embedder = SentenceEmbedder.reload(params.model_path, params)

# reload langs from pretrained model
params.n_langs = embedder.pretrain_params['n_langs']
params.id2lang = embedder.pretrain_params['id2lang']
params.lang2id = embedder.pretrain_params['lang2id']

# create xnli instance
scores = {}
xnli = XNLI(embedder, scores, params)

# get attention scores
sentences, scores = xnli.save_attention_scores()

# for sentence in sentences['en'].T:
#     s = ''
#     for word in sentence:
#         s += embedder.model.dico.id2word[int(word)] + ' '
#     print(s)
#     print('')
#
# for sentence in sentences['es'].T:
#     s = ''
#     for word in sentence:
#         s += embedder.model.dico.id2word[int(word)] + ' '
#     print(s)
#     print('')

# create images
t_layer = 11
direction = 'backward'
for lang in ['en', 'es']:
    # get words from sentence
    for s_ind, sentence in enumerate(sentences[lang].T):
        s = []
        for word in sentence:
            token = embedder.model.dico.id2word[int(word)]
            if token == '<pad>':
                continue
            s.append(token)

        for w_ind, word in enumerate(s):
            for head in range(scores[lang][t_layer].shape[1]):
                plt.figure()
                fig, axs = plt.subplots(1, 2)
                # cbar_ax = fig.add_axes([.91, .3, .015, .4])

                if direction == 'backward':
                    s_score = scores[lang][t_layer][s_ind, head, :, w_ind]
                elif direction == 'forward':
                    s_score = scores[lang][t_layer][s_ind, head, w_ind, :]

                axs[0].set_title('Attention Scores')
                sn.heatmap(np.expand_dims(s_score[:len(s)], axis=1), annot=np.array([s]).T, fmt='', ax=axs[0], yticklabels=False, xticklabels=False, cbar_kws = dict(use_gridspec=False,location="left"))

                axs[1].set_title('Selected Word')
                word_choice = np.zeros(len(s))
                word_choice[w_ind] += 0.5
                sn.heatmap(np.expand_dims(word_choice, axis=1), annot=np.array([s]).T, fmt='', ax=axs[1],
                           yticklabels=False, xticklabels=False, cbar=False)

                path = ('./dumped/attention_weight_vis/' + params.exp_name +
                            '/' + lang +
                            '/sentence_' + str(s_ind))
                os.makedirs(path, exist_ok=True)
                plt.savefig(path + '/word_' + str(w_ind) + '_head_' + str(head) + '.png', dpi=300)
                plt.close()


