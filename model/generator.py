import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable
from model.modules import *
import pdb 

class Generator(nn.Module):
    def __init__(self, d_model, vocab, W=None):
        super(Generator, self).__init__()
        if W is not None:
            self.proj = W
            self.shared_W = True
        else:
            self.proj = nn.Linear(d_model, vocab)
            self.shared_W = False
        
    #def forward(self, x):
    def forward(self, ft, batch, args, ft_key='decoded_text'):
        x = ft[ft_key]
        if hasattr(self, 'shared_W') and self.shared_W:
            out = x.matmul(self.proj.transpose(1,0))
            return F.log_softmax(out, dim=-1)
        else:
            return F.log_softmax(self.proj(x), dim=-1)
        
class PointerGenerator(nn.Module):
    def __init__(self, d_model, vocab_gen, pointer_attn):
        super(PointerGenerator, self).__init__()
        self.vocab_gen = vocab_gen
        self.pointer_gen_W = nn.Linear(d_model*3, 1) 
        self.pointer_attn = pointer_attn 

    #def forward(self, logits, ft, batch, args):
    def forward(self, ft, batch, args):
        #vocab_attn =  logits.matmul(self.vocab_gen.transpose(1,0))
        logits = ft['decoded_text']
        vocab_attn =  logits.matmul(self.vocab_gen.transpose(1,0))
        p_vocab = F.softmax(vocab_attn, dim = -1)
        
        if args.ptr_ft == 'query':
            text = batch.query
            encoded_text = ft['encoded_query']
            text_mask = batch.query_mask
        elif args.ptr_ft == 'his':
            text = batch.his
            encoded_text = ft['encoded_his']
            text_mask = batch.his_mask
        elif args.ptr_ft == 'cap':
            text = batch.cap
            encoded_text = ft['encoded_cap']
            text_mask = batch.cap_mask
        elif args.ptr_ft == 'query+cap':
            text = torch.cat([batch.query, batch.cap], dim=1)
            encoded_text = torch.cat([ft['encoded_query'], ft['encoded_cap']], dim=1)
            text_mask = torch.cat([batch.query_mask, batch.cap_mask], dim=2)
        encoded_in = ft['encoded_tgt']
        
        if args.mask_unk:
            text_mask = text_mask & (text != 0).unsqueeze(-2)
        
        self.pointer_attn(logits, encoded_text, encoded_text, text_mask)
        pointer_attn = self.pointer_attn.attn.squeeze(1)
        #p_vocab = F.softmax(vocab_attn, dim = -1)
        
        text_index = text.unsqueeze(1).expand_as(pointer_attn)
        p_text_ptr = torch.zeros(p_vocab.size()).cuda()
        p_text_ptr.scatter_add_(2, text_index, pointer_attn)
                        
        expanded_pointer_attn = pointer_attn.unsqueeze(-1).repeat(1, 1, 1, encoded_text.shape[-1])
        text_vec = (encoded_text.unsqueeze(1).expand_as(expanded_pointer_attn) * expanded_pointer_attn).sum(2)
        p_gen_vec = torch.cat([logits, text_vec, encoded_in], -1)
        vocab_pointer_switches = nn.Sigmoid()(self.pointer_gen_W(p_gen_vec)).expand_as(p_text_ptr)
        p_out = (1 - vocab_pointer_switches) * p_text_ptr + vocab_pointer_switches * p_vocab
            
        return torch.log(p_out)

class MultiPointerGenerator(nn.Module):
    def __init__(self, d_model, vocab_gen, pointer_attn, nb_pointer_ft):
        super(MultiPointerGenerator, self).__init__()
        self.vocab_gen = vocab_gen
        self.pointer_gen_W = nn.Linear(d_model*(nb_pointer_ft+2), nb_pointer_ft+1) 
        self.pointer_attn = pointer_attn 

    def forward(self, ft, batch, args):
        logits = ft['decoded_text']
        vocab_attn =  logits.matmul(self.vocab_gen.transpose(1,0))
        p_vocab = F.softmax(vocab_attn, dim = -1)

        ptr_ft_ls = args.ptr_ft.split(',')
        p_text_ptr_ls = []
        encoded_in = ft['encoded_tgt']
        p_gen_vec = [logits, encoded_in]
        for idx, ptr_ft in enumerate(ptr_ft_ls):
            if ptr_ft == 'query':
                text = batch.query
                encoded_text = ft['encoded_query']
                text_mask = batch.query_mask
            elif ptr_ft == 'his':
                text = batch.his
                encoded_text = ft['encoded_his']
                text_mask = batch.his_mask
            elif ptr_ft == 'cap':
                text = batch.cap
                encoded_text = ft['encoded_cap']
                text_mask = batch.cap_mask
            if args.mask_unk:
                text_mask = text_mask & (text != 0).unsqueeze(-2)
                
            self.pointer_attn[idx](logits, encoded_text, encoded_text, text_mask)
            pointer_attn = self.pointer_attn[idx].attn.squeeze(1)

            text_index = text.unsqueeze(1).expand_as(pointer_attn)
            p_text_ptr = torch.zeros(p_vocab.size()).cuda()
            p_text_ptr.scatter_add_(2, text_index, pointer_attn)
            p_text_ptr_ls.append(p_text_ptr)
            
            expanded_pointer_attn = pointer_attn.unsqueeze(-1).repeat(1, 1, 1, encoded_text.shape[-1])
            text_vec = (encoded_text.unsqueeze(1).expand_as(expanded_pointer_attn) * expanded_pointer_attn).sum(2)
            p_gen_vec.append(text_vec)
            
        p_gen_vec = torch.cat(p_gen_vec, -1)
        vocab_pointer_p = F.softmax(self.pointer_gen_W(p_gen_vec), dim=-1)
        p_out = 0.0
        for idx, ptr_ft in enumerate(ptr_ft_ls):
            p_out += vocab_pointer_p[:,:,idx].unsqueeze(-1).expand_as(p_text_ptr) * p_text_ptr_ls[idx]
        p_out += vocab_pointer_p[:,:,-1].unsqueeze(-1).expand_as(p_vocab) * p_vocab
        return torch.log(p_out)