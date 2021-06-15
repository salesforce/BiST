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
import re
import io
import random 
from collections import Counter
from nltk.util import ngrams
import torch 
import torch.utils.data as Data
from tqdm import tqdm
from data.dataset import *
from data.data_utils import *

def get_vocabulary(dataset_file, cutoff=0, include_caption='none', ptr_gen=0):
    vocab = {'<unk>':0, '<blank>':1, '<sos>':2, '<eos>':3}
    dialog_data = json.load(open(dataset_file, 'r'))
    word_freq = {}
    for dialog in dialog_data['dialogs']:
        if include_caption == 'caption' or include_caption == 'summary' or include_caption == 'caption,summary':
            if include_caption == 'caption' or include_caption == 'summary':
                caption = dialog[include_caption]
            else:
                caption = dialog['caption'] + dialog['summary']
            for word in caption.split():
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        for key in ['question', 'answer']:
            for turn in dialog['dialog']:
                for word in turn[key].split():
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
    if ptr_gen:
        vocab = {'<unk>':0, '<blank>':1, '<sos>':2, '<eos>':3}
        for word, freq in word_freq.items():
            vocab[word] = len(vocab) 
        print("Pointer Gen {} --> {} words".format(ptr_gen, len(vocab)))
    else:
        cutoffs = range(cutoff+1)
        for cutoff in cutoffs:
            vocab = {'<unk>':0, '<blank>':1, '<sos>':2, '<eos>':3}
            for word, freq in word_freq.items():
                if freq > cutoff:
                    vocab[word] = len(vocab) 
            print("{} words for cutoff {}".format(len(vocab), cutoff))
    return vocab

# load text data
def load(fea_types, fea_path, dataset_file, vocab, include_caption='none', separate_caption=False, max_history_length=-1, merge_source=False, undisclosed_only=False, skip=0):
    dialog_data = json.load(open(dataset_file, 'r'))
    dialog_list = []
    vid_set = set()
    qa_id = 0
    for dial_idx, dialog in enumerate(dialog_data['dialogs']):
        #if dial_idx == 10: 
        #    break 
        if include_caption == 'caption' or include_caption == 'summary':
            caption = words2ids(dialog[include_caption], vocab)
        elif include_caption == 'caption,summary':
            caption = words2ids(dialog['caption'] + dialog['summary'], vocab)
        else:
            caption = np.array([vocab['<blank>']], dtype=np.int32)
        questions = [words2ids(d['question'], vocab) for d in dialog['dialog']]
        answers = [words2ids(d['answer'], vocab) for d in dialog['dialog']]
        qa_pair = [np.concatenate((q,a)).astype(np.int32) for q,a in zip(questions, answers)]
        vid = dialog['image_id']
        vid_set.add(vid)
        if undisclosed_only:
            it = range(len(questions)-1,len(questions))
        else:
            it = range(len(questions))
        for n in it:
            if undisclosed_only:
                assert dialog['dialog'][n]['answer'] == '__UNDISCLOSED__'
            if (include_caption == 'caption' or include_caption == 'summary' or include_caption == 'caption,summary') and separate_caption:
                history = [np.array([vocab['<blank>']], dtype=np.int32)]
            else:
                history = [caption]
            if max_history_length > 0: 
                start_turn_idx = max(0, n - max_history_length)
            else:
                start_turn_idx = 0 
            for m in range(start_turn_idx, n):
                history = np.append(history, qa_pair[m])
            if type(history) == list: #only including caption i.e. no dialogue history 
                history = history[0]
            question = questions[n]
            if merge_source:
                question = np.concatenate((caption, history, question))
            answer_in = answers[n][:-1]
            answer_out = answers[n][1:]
            item = [vid, qa_id, history, question, answer_in, answer_out]
            if (include_caption == 'caption' or include_caption == 'summary' or include_caption == 'caption,summary') and separate_caption:
                item.append(caption)
            dialog_list.append(item)
            qa_id += 1
        if ('train_test' in dataset_file or 'valid_test' in dataset_file or 'test_test' in dataset_file) and qa_id>100: break
    data = {'dialogs': dialog_list, 'vocab': vocab, 'features': [], 
            'original': dialog_data}
    if fea_types is not None and fea_types[0] != 'none':
        for ftype in fea_types:
            if ftype == 'none': 
                data['features'].append(None)
                continue
            basepath = fea_path.replace('<FeaType>', ftype)
            features = {}
            for vid in tqdm(vid_set, total=len(vid_set)):
                filepath = basepath.replace('<ImageID>', vid)
                if 'rgb' in ftype:
                    feature = np.load(filepath, allow_pickle=True)[::skip]
                    shape = feature.shape
                elif 'st' in ftype: # spatio-temporal feature
                    feature = None 
                    shape=[1]
                else:
                    feature = None
                    shape=[1]
                features[vid] = (filepath, shape[0], feature)
            data['features'].append(features)
    else:
        data['features'] = None 
    return data 

def create_dataset(data, batch_size, shuffle, include_caption='none', separate_caption=False, cut_a=False, num_workers=0):
    out = {}
    keys = ['vid', 'qa_id', 'history', 'question', 'answer_in', 'answer_out', 'caption', 'features']
    fts = data['features']
    for key in keys:
        out[key] = []
    for dialog in data['dialogs']:
        out['vid'].append(dialog[0])
        out['qa_id'].append(dialog[1])
        out['history'].append(dialog[2])
        out['question'].append(dialog[3])
        out['answer_in'].append(dialog[4])
        out['answer_out'].append(dialog[5])
        
        if (include_caption == 'caption' or include_caption == 'summary' or include_caption == 'caption,summary') and separate_caption:
            out['caption'].append(dialog[6])
        if fts is not None:
            temp = []
            for ft in fts:
                if ft is None:
                    temp.append(None)
                else:
                    temp.append(ft[dialog[0]])
            out['features'].append(temp)
    dataset = Dataset(out, cut_a)         
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  collate_fn=collate_fn,
                                                  num_workers=num_workers,
                                                  pin_memory=True)
    return data_loader, len(out['vid'])
    
def feature_shape(data):
    dims = []
    for features in data["features"]:
        sample_feature = list(features.values())[0]
        if isinstance(sample_feature, tuple):
	        dims.append(np.load(sample_feature[0], allow_pickle=True).shape[-1])
        else:
            dims.append(sample_feature.shape[1])
    return dims
