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
import torch.utils.data as Data
from torch.autograd import Variable
from data.data_utils import *

class Dataset(Data.Dataset):
    def __init__(self, data_info, cut_a):
        self.vid = data_info['vid']
        self.qa_id = data_info['qa_id']
        self.history = data_info['history']
        self.question = data_info['question']
        self.answer_in = data_info['answer_in']
        self.answer_out = data_info['answer_out']
        self.caption = data_info['caption']
        self.features = data_info['features']
        self.num_total_seqs = len(data_info['vid'])
        self.cut_a = cut_a

    def __getitem__(self, index): 
        answer_in = self.answer_in[index]
        answer_out = self.answer_out[index]
        if self.cut_a:
            pr = np.random.uniform()
            if pr >= 0.5: 
                end_idx = np.random.choice(range(1, len(answer_in)), 1)[0]
                answer_out = np.concatenate((answer_in[1:end_idx],[answer_in[end_idx]]))
                answer_in = answer_in[:end_idx]
        if len(self.features) == 0: 
            features = None
        else:
            features = self.features[index]

        item_info = {
            'vid':self.vid[index], 
            'qa_id': self.qa_id[index],
            'history': self.history[index],
            'question': self.question[index],
            'answer_in': answer_in,
            'answer_out': answer_out,
            'caption': self.caption[index] if len(self.caption)>0 else None,
            'features': features
            }
        return item_info
    
    def __len__(self):
        return self.num_total_seqs

class Batch:
    def __init__(self, query, his, fts, cap, trg, trg_y, pad, vids, qa_ids, cuda=False, audio_fts=None):
        self.vids = vids
        self.qa_ids = qa_ids
        self.cuda = cuda
        self.query = self.to_cuda(query)
        self.his = self.to_cuda(his)
        self.query_mask = self.to_cuda((query != pad).unsqueeze(-2))
        self.his_mask  = self.to_cuda((his != pad).unsqueeze(-2))
        
        self.temporal_ft = None
        self.spatial_ft = None 
        self.fts = None 
        self.spatial_mask = None 
        self.temporal_mask = None
        self.fts_mask = None 
        if fts is not None and len(fts)!=0:
            if len(fts[0].shape)==4: # Full scale spatio-temporal features
                self.fts = self.to_cuda(torch.from_numpy(fts[0]).float())
                self.spatial_mask = self.to_cuda((self.fts.sum(1).sum(-1)!=0).unsqueeze(-2))
                self.temporal_mask = self.to_cuda((self.fts.sum(2).sum(-1)!=0).unsqueeze(-2))
            else:
                self.fts = self.to_cuda(torch.from_numpy(fts[0]).float())
                self.fts_mask = self.to_cuda((self.fts.sum(-1)!=0).unsqueeze(-2))
        self.audio_fts = None
        self.audio_mask = None 
        if audio_fts is not None and len(audio_fts)!=0:
            self.audio_fts = self.to_cuda(torch.from_numpy(audio_fts[0]).float())                                        
            self.audio_mask = self.to_cuda((self.audio_fts.sum(-1)!=0).unsqueeze(-2))
        self.cap = None
        self.cap_mask = None
        if cap is not None:
            self.cap = self.to_cuda(cap)
            self.cap_mask = self.to_cuda((cap != pad).unsqueeze(-2))

        self.trg = self.to_cuda(trg)
        self.trg_y = self.to_cuda(trg_y)
        self.trg_mask = self.to_cuda(self.make_std_mask(self.trg, pad))
        self.trg_mean_mask = self.to_cuda((self.trg_y != pad))
        self.ntokens = (self.trg_y != pad).data.sum()
        self.qntokens = (self.query != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask    

    def move_to_cuda(self):
        self.query = self.query.to('cuda', non_blocking=True)
        self.his = self.his.to('cuda', non_blocking=True)
        self.query_mask = self.query_mask.to('cuda', non_blocking=True)
        self.his_mask = self.his_mask.to('cuda', non_blocking=True)
        if self.fts is not None:
            self.fts = self.fts.to('cuda', non_blocking=True)
            if self.fts_mask is not None:
                self.fts_mask = self.fts_mask.to('cuda', non_blocking=True)
            else:
                self.spatial_mask = self.spatial_mask.to('cuda', non_blocking=True)
                self.temporal_mask = self.temporal_mask.to('cuda', non_blocking=True)
        if self.audio_fts is not None:
            self.audio_fts = self.audio_fts.to('cuda', non_blocking=True)
            self.audio_mask = self.audio_mask.to('cuda', non_blocking=True)
        if self.cap is not None:
            self.cap = self.cap.to('cuda', non_blocking=True)
            self.cap_mask = self.cap_mask.to('cuda', non_blocking=True)
        self.trg = self.trg.to('cuda', non_blocking=True)
        self.trg_y = self.trg_y.to('cuda', non_blocking=True)
        self.trg_mask = self.trg_mask.to('cuda', non_blocking=True)
    
    def to_cuda(self, tensor):
        if self.cuda: return tensor.cuda()
        return tensor 
    
def collate_fn(data):
    def pad_seq(seqs, pad_token):
        max_length = max([s.shape[0] for s in seqs])
        output = []
        for seq in seqs:
            result = np.ones(max_length, dtype=seq.dtype)*pad_token
            result[:seq.shape[0]] = seq 
            output.append(result)
        return output 

    def prepare_data(seqs):
        return torch.from_numpy(np.asarray(seqs)).long()
            
    def load_np(filepath):
        feature = np.load(filepath, allow_pickle=True)
        if len(feature.shape) == 2:
            return feature
        else:
            return feature.reshape((feature.shape[0], -1, feature.shape[-1]))

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
    
    if item_info['features'][0] != None:
        features = []
        audio_features = []
        for f_idx in range(len(item_info['features'][0])): 
            if item_info['features'][0][f_idx] is None:
                features.append(None)
            else:
                fea_ls = [load_np(fi[f_idx][0]) for fi in item_info['features']]
                x_len = max([i.shape[0] for i in fea_ls]) 
                n_seqs = len(fea_ls)
                if len(fea_ls[0].shape) == 3: #Full scale spatio-temporal feature
                    x_batch = np.zeros((n_seqs, x_len, 
                                        fea_ls[0].shape[1], 
                                        fea_ls[0].shape[2]),dtype=np.float32)        
                else:
                    x_batch = np.zeros((n_seqs, x_len, fea_ls[0].shape[-1]),dtype=np.float32)
                for j, fea in enumerate(fea_ls):
                    x_batch[j, :len(fea)] = fea
                if 'vggish' in item_info['features'][0][f_idx][0]:
                    audio_features.append(x_batch)
                else:
                #if True:
                    features.append(x_batch)
    else:
        features = None
        audio_features = None
    
    h_batch = prepare_data(pad_seq(item_info['history'], 1))
    q_batch = prepare_data(pad_seq(item_info['question'], 1))
    a_batch_in = prepare_data(pad_seq(item_info['answer_in'], 1))
    a_batch_out = prepare_data(pad_seq(item_info['answer_out'], 1))
    if item_info['caption'][0] is None:
        c_batch = None
    else:
        c_batch = prepare_data(pad_seq(item_info['caption'], 1))
    
    batch = Batch(q_batch, h_batch, features, c_batch, a_batch_in, a_batch_out, 1, item_info['vid'], item_info['qa_id'], audio_fts=audio_features)

    return batch
