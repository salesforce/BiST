import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy, time
import pdb 
import re
from collections import Counter
from nltk.util import ngrams
from data.data_utils import *

def oracle_decode(model, batch):
    video_features, video_features_mask, cap, cap_mask, his, his_st, his_mask, query, query_mask = batch.fts, batch.fts_mask, batch.cap, batch.cap_mask, batch.his, batch.his_st, batch.his_mask, batch.query, batch.query_mask
    
    his_memory, cap_memory, query_memory, encoded_vid_features, ae_encoded_ft = encode(model, his, his_st, his_mask, cap, cap_mask, query, query_mask, video_features, video_features_mask)

    out = model.decode(encoded_vid_features, his_memory, cap_memory, query_memory,
                      video_features_mask, his_mask, cap_mask, query_mask, 
                      batch.trg, batch.trg_mask, 
                      ae_encoded_ft)
    if type(out) == list or type(out) == tuple:
        prob = model.generator(out[0])[0]
    else:
        prob = model.generator(out)[0]
    _, out = torch.max(prob, dim = 1)
    return out


def greedy_decode(model, batch, max_len, start_symbol, pad_symbol):
    video_features, video_features_mask, cap, cap_mask, his, his_st, his_mask, query, query_mask = batch.fts, batch.fts_mask, batch.cap, batch.cap_mask, batch.his, batch.his_st, batch.his_mask, batch.query, batch.query_mask
    
    his_memory, cap_memory, query_memory, encoded_vid_features, ae_encoded_ft = encode(model, his, his_st, his_mask, cap, cap_mask, query, query_mask, video_features, video_features_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).type_as(query.data)

    for i in range(max_len-1):
        cap2res_mask = None
        out = model.decode(encoded_vid_features, his_memory, cap_memory, query_memory,
                          video_features_mask, his_mask, cap_mask, query_mask, 
                          Variable(ys), 
                          Variable(subsequent_mask(ys.size(1)).type_as(query.data)),
                          ae_encoded_ft)
        if type(out) == list or type(out) == tuple:
            #prob = 0
            #for idx, o in enumerate(out):
            #    prob += model.generator[idx](o[:,-1])
            prob = model.generator(out[0][:,-1])
        else:
            prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(query.data).fill_(next_word)], dim=1)
    return ys

def beam_search_decode(model, batch, max_len, start_symbol, unk_symbol, end_symbol, pad_symbol, beam=5, penalty=1.0, nbest=5, min_len=1, train_args=None, dec_eos=False):
    ft = model.encode(batch)
    ds = torch.ones(1, 1).fill_(start_symbol).long()
    hyplist=[([], 0., ds)]
    best_state=None
    comp_hyplist=[]
    for l in range(max_len): 
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            batch.trg = Variable(st).cuda()
            batch.trg_mask = Variable(subsequent_mask(st.size(1)).long()).cuda()
            batch.trg_mean_mask = torch.ones(batch.trg.shape).long().cuda()
            ft = model.decode(batch, ft)
            ft['decoded_text'] = ft['decoded_text'][:,-1].unsqueeze(1)
            ft['encoded_tgt'] = ft['encoded_tgt'][:,-1].unsqueeze(1) 

            logp = model.generator(ft, batch, train_args)
            lp_vec = logp.cpu().data.numpy() + lp 
            lp_vec = np.squeeze(lp_vec)
            if l >= min_len:
                new_lp = lp_vec[end_symbol] + penalty * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp: 
                    best_state = new_lp
            count = 1 
            for o in np.argsort(lp_vec)[::-1]:
                if dec_eos and (o == unk_symbol):
                    continue 
                if not dec_eos and (o == unk_symbol or o == end_symbol):
                    continue 
                new_lp = lp_vec[o]
                if len(new_hyplist) == beam:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = torch.cat([st, torch.ones(1,1).long().fill_(int(o))], dim=1)
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                    else:
                        break
                else: 
                    new_st = torch.cat([st, torch.ones(1,1).long().fill_(int(o))], dim=1)
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == beam:
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                count += 1
        hyplist = new_hyplist 
            
    if len(comp_hyplist) > 0: 
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
        return maxhyps, best_state
    else:
        return [([], 0)], None

def ensemble_beam_search_decode(models, batch, max_len, start_symbol, unk_symbol, end_symbol, pad_symbol, beam=5, penalty=1.0, nbest=5, min_len=1, train_args=None, dec_eos=False):
    video_features, video_features_mask, cap, cap_mask, his, his_st, his_mask, query, query_mask = batch.fts, batch.fts_mask, batch.cap, batch.cap_mask, batch.his, None, batch.his_mask, batch.query, batch.query_mask
    
    encoded = {}
    count = 0
    for key, model in models.items():
        #his_memory, cap_memory, query_memory, encoded_vid_features, ae_encoded_ft
        vid_fts = [video_features[count]] if video_features[count] is not None else None
        vid_fts_mask = [video_features_mask[count]] if video_features_mask[count] is not None else None 
        encoded[key] = encode(model, his, his_st, his_mask, cap, cap_mask, query, query_mask, vid_fts, vid_fts_mask)
        count += 1
        
    his_memory, cap_memory, query_memory, encoded_vid_features, ae_encoded_ft = encoded['resnext101'] if 'resnext101' in encoded else encoded['resnext_101']
    ds = torch.ones(1, 1).fill_(start_symbol).type_as(query.data)
    hyplist=[([], 0., ds)]
    best_state=None
    comp_hyplist=[]
    for l in range(max_len): 
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            cap2res_mask = None
            logp = 0
            count = 0
            for key, model in models.items():
                vid_fts_mask = [video_features_mask[count]]
                his_memory, cap_memory, query_memory, encoded_vid_features, ae_encoded_ft = encoded[key]
                if hasattr(model, 'ptr_gen') and model.ptr_gen:
                    ft = {}
                    ft['encoded_query'] = query_memory
                    ft['encoded_cap'] = cap_memory
                    ft['encoded_his'] = his_memory
                    output, _, encoded_tgt =  model.decode(encoded_vid_features, his_memory, cap_memory, query_memory, vid_fts_mask, his_mask, cap_mask, query_mask,  Variable(st), Variable(subsequent_mask(st.size(1)).type_as(query.data)), ae_encoded_ft)
                    ft['encoded_tgt'] = encoded_tgt[:,-1].unsqueeze(1)
                else:
                    output = model.decode(encoded_vid_features, his_memory, cap_memory, query_memory,
                                      vid_fts_mask, his_mask, cap_mask, query_mask,
                                      Variable(st),
                                      Variable(subsequent_mask(st.size(1)).type_as(query.data)),
                                      ae_encoded_ft)
                if type(output) == tuple or type(output) == list:
                    logp += model.generator(output[0][:, -1])
                else:
                    if hasattr(model, 'ptr_gen') and model.ptr_gen:
                        logp += model.generator(output[:, -1].unsqueeze(1), ft, batch, train_args)
                    else:
                        logp += model.generator(output[:, -1])
                count += 1
            lp_vec = logp.cpu().data.numpy() + lp 
            lp_vec = np.squeeze(lp_vec)
            if l >= min_len:
                new_lp = lp_vec[end_symbol] + penalty * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp: 
                    best_state = new_lp
            count = 1 
            for o in np.argsort(lp_vec)[::-1]:
                if dec_eos and (o == unk_symbol):
                    continue 
                if not dec_eos and (o == unk_symbol or o == end_symbol):
                    continue 
                new_lp = lp_vec[o]
                if len(new_hyplist) == beam:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = torch.cat([st, torch.ones(1,1).type_as(query.data).fill_(int(o))], dim=1)
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                    else:
                        break
                else: 
                    new_st = torch.cat([st, torch.ones(1,1).type_as(query.data).fill_(int(o))], dim=1)
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == beam:
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                count += 1
        hyplist = new_hyplist 
            
    if len(comp_hyplist) > 0: 
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
        return maxhyps, best_state
    else:
        return [([], 0)], None