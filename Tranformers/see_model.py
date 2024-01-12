
import torch
import torch.nn as nn
import pickle
from pretrain_trfm import *
from build_vocab import WordVocab
# from test_vocab import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# test_vocab()
# WordVocab

with open('../data/raw/bace.csv', "r", encoding='utf-8') as f:
    vocab = WordVocab(f, max_size=None, min_freq=1)


# vocab = WordVocab.load_vocab('data/vocab.pkl')
trfm = TrfmSeq2seq(45, 256, 45, 3)
if device.type == 'cpu':
    map_location = torch.device('cpu')

trfm.load_state_dict(torch.load('./pre_trained_model/trfm_12_23000.pkl', map_location=map_location))


##

letmeknow=1



