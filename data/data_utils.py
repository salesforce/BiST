import copy
import logging
import sys
import time
import os
import six
import pickle
import json
import numpy as np
import pdb 
from tqdm import tqdm
import torch 

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def get_npy_shape(filename):
    with open(filename, 'rb') as f:
        if filename.endswith('.pkl'):
            shape = pickle.load(f).shape
        else:
            pdb.set_trace()
            major, minor = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    return shape

def words2ids(str_in, vocab):
    words = str_in.split()
    sentence = np.ndarray(len(words)+2, dtype=np.int32)
    sentence[0]=vocab['<sos>']
    for i,w in enumerate(words):
        if w in vocab:
            sentence[i+1] = vocab[w]
        else:
            sentence[i+1] = vocab['<unk>']
    sentence[-1]=vocab['<eos>']
    return sentence