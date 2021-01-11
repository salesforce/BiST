import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable
import pdb
from model.modules import *

class MultimodalDecoderLayer12(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, dropout, args):
        super(MultimodalDecoderLayer12, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = ff
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn+1)
        self.args = args 
                
    def forward(self, b, ft, x):
        s_x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        h_x = self.sublayer[1](s_x, lambda s_x: self.attn[1](s_x, ft['encoded_his'], ft['encoded_his'], b.his_mask)) 
        q_x = self.sublayer[2](h_x, lambda h_x: self.attn[2](h_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
        
        in_x = q_x
        count = 3
        if self.args.nb_venc_blocks>0 and self.args.nb_cenc_blocks>0 and (hasattr(self.args, 'enc_vc_combine') and self.args.enc_vc_combine != 'none'):
            in_x = self.sublayer[count](in_x, lambda in_x: self.attn[count](in_x, ft['encoded_ft'], ft['encoded_ft'], b.query_mask))
            count += 1
        else:
            if self.args.include_caption != 'none': 
                if self.args.nb_cenc_blocks>0:
                    in_x = self.sublayer[count](in_x, lambda in_x: self.attn[count](in_x, ft['cap_ft'], ft['cap_ft'], b.query_mask))
                else:
                    in_x = self.sublayer[count](in_x, lambda in_x: self.attn[count](in_x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
                count += 1
            if self.args.nb_venc_blocks>0:
                if self.args.enc_st_combine == 'none':
                    if self.args.dec_st_combine == 'seq':
                        if self.args.s2t:
                            in_x = self.sublayer[count](in_x, lambda in_x: self.attn[count](in_x, ft['temporal_ft'], ft['temporal_ft'], b.query_mask))
                            count += 1
                        if self.args.t2s:
                            in_x = self.sublayer[count](in_x, lambda in_x: self.attn[count](in_x, ft['spatial_ft'], ft['spatial_ft'], b.query_mask))
                            count += 1
                    else:
                        temporal_x = self.sublayer[count](in_x, lambda in_x: self.attn[count](in_x, ft['temporal_ft'], ft['temporal_ft'], b.query_mask))
                        count += 1
                        spatial_x = self.sublayer[count](in_x, lambda in_x: self.attn[count](in_x, ft['spatial_ft'], ft['spatial_ft'], b.query_mask))
                        count += 1
                        in_x = temporal_x + spatial_x
                else:
                    in_x = self.sublayer[count](in_x, lambda in_x: self.attn[count](in_x, ft['spatiotemporal_ft'], ft['spatiotemporal_ft'], b.query_mask))
                    count += 1
            if hasattr(self.args, "nb_aenc_blocks") and self.args.nb_aenc_blocks>0:
                in_x = self.sublayer[count](in_x, lambda in_x: self.attn[count](in_x, ft['audio_ft'], ft['audio_ft'], b.query_mask))
                count += 1
        out_x = self.sublayer[count](in_x, self.ff)

        return out_x
    
class MultimodalDecoder8(nn.Module):
    def __init__(self, v_layer, c_layer, a_layer, layer, venc_N, cenc_N, aenc_N, N, args):
        super(MultimodalDecoder8, self).__init__()
        self.layers = clones(layer, N)
        self.N=N
        self.v_N=venc_N
        self.c_N=cenc_N
        self.a_N=aenc_N
        self.norm = LayerNorm(layer.size)
        self.args = args
        
        if self.v_N>0:
            self.v_layers = clones(v_layer, self.v_N)
            if self.args.enc_st_combine == 'none':
                self.spatial_out_norm = LayerNorm(v_layer.size)
                self.temporal_out_norm = LayerNorm(v_layer.size)
            elif self.args.enc_st_combine not in ['early_sum', 'early_dyn']:
                self.out_norm = LayerNorm(v_layer.size)
                if self.args.enc_st_combine == 'dyn':
                    self.st_combine_W = nn.Linear(v_layer.size*3, 1)
        
        if self.c_N>0:
            self.c_layers = clones(c_layer, self.c_N) 
            self.cap_out_norm = LayerNorm(c_layer.size)
            
        if self.a_N>0:
            self.a_layers = clones(a_layer, self.a_N) 
            self.a_out_norm = LayerNorm(a_layer.size)
            
        if self.v_N>0 and self.args.enc_vc_combine=='dyn':
            if self.args.enc_st_combine in ['sum', 'dyn'] and self.args.s2t and self.args.t2s:
                self.vc_combine_W = nn.Linear(v_layer.size*3, 1)                
            else:
                factor = 1
                if self.args.include_caption != 'none':
                    factor += 1
                if self.args.t2s:
                    factor += 1
                if self.args.s2t:
                    factor += 1
                if self.a_N>0:
                    factor += 1
                 
                self.vc_combine_W = nn.Linear(v_layer.size*factor, factor-1)

    def forward(self, b, ft, x):
        in_ft = {}
        in_ft['t2s'] = ft['encoded_query']
        in_ft['s2t'] = ft['encoded_query']
        in_ft['audio'] = ft['encoded_query']
        in_ft['cap'] = ft['encoded_query']
        count = 0
        for layer in self.layers:
            if self.v_N>0:
                in_ft = self.v_layers[count](in_ft, ft, b)
                if self.args.enc_st_combine == 'sum' and self.args.s2t and self.args.t2s:
                    ft['spatiotemporal_ft'] = self.out_norm(in_ft['s2t']+in_ft['t2s'])
                elif self.args.enc_st_combine == 'dyn' and self.args.s2t and self.args.t2s:
                    temp = torch.cat([ft['encoded_query'], in_ft['s2t'], in_ft['t2s']], dim=-1)
                    combine_score = nn.Sigmoid()(self.st_combine_W(temp))
                    ft['spatiotemporal_ft'] = self.out_norm(combine_score*in_ft['s2t'] + (1-combine_score)*in_ft['t2s'])
                elif self.args.enc_st_combine in ['early_sum', 'early_dyn'] and self.args.s2t and self.args.t2s:
                    ft['spatiotemporal_ft'] = in_ft['s2t']
                else:
                    if self.args.s2t:
                        ft['temporal_ft']  = self.temporal_out_norm(in_ft['s2t'])
                    if self.args.t2s:
                        ft['spatial_ft'] = self.spatial_out_norm(in_ft['t2s']) 
            if self.c_N>0:
                in_ft = self.c_layers[count](in_ft, ft, b)          
                ft['cap_ft'] = self.cap_out_norm(in_ft['cap']) 
            if self.a_N>0:
                in_ft = self.a_layers[count](in_ft, ft, b)          
                ft['audio_ft'] = self.a_out_norm(in_ft['audio']) 
                
            if self.v_N>0 and self.c_N>0 and (hasattr(self.args, 'enc_vc_combine') and self.args.enc_vc_combine == 'sum'):
                if self.args.enc_st_combine in ['sum', 'dyn'] and self.args.s2t and self.args.t2s:
                    ft['encoded_ft'] = ft['spatiotemporal_ft'] + ft['cap_ft']                
                else:
                    ft['encoded_ft'] = ft['temporal_ft'] + ft['spatial_ft'] + ft['cap_ft']        
            elif self.v_N>0 and self.c_N>0 and (hasattr(self.args, 'enc_vc_combine') and self.args.enc_vc_combine == 'dyn'):
                if self.args.enc_st_combine in ['sum', 'dyn'] and self.args.s2t and self.args.t2s:
                    temp = torch.cat([ft['encoded_query'], ft['spatiotemporal_ft'], ft['cap_ft']], dim=-1)
                    combine_score = nn.Sigmoid()(self.vc_combine_W(temp))
                    ft['encoded_ft'] = combine_score*ft['spatiotemporal_ft'] + (1-combine_score)*ft['cap_ft']                
                else:
                    temp = torch.cat([ft['encoded_query'], ft['cap_ft']], dim=-1)
                    if self.args.t2s:
                        temp = torch.cat([temp, ft['spatial_ft']], dim=-1)
                    if self.args.s2t:
                        temp = torch.cat([temp, ft['temporal_ft']], dim=-1)
                    if self.a_N>0:
                        temp = torch.cat([temp, ft['audio_ft']], dim=-1)
                    combine_score = F.softmax((self.vc_combine_W(temp)), dim=-1)
                    if self.args.t2s and self.args.s2t:
                        ft['encoded_ft'] = combine_score[:,:,0].unsqueeze(-1)*ft['temporal_ft'] + \
                            combine_score[:,:,1].unsqueeze(-1)*ft['spatial_ft'] + \
                            combine_score[:,:,2].unsqueeze(-1)*ft['cap_ft'] 
                    elif not self.args.t2s:
                        ft['encoded_ft'] = combine_score[:,:,0].unsqueeze(-1)*ft['temporal_ft'] + \
                            combine_score[:,:,1].unsqueeze(-1)*ft['cap_ft'] 
                    elif not self.args.s2t:
                        ft['encoded_ft'] = combine_score[:,:,0].unsqueeze(-1)*ft['spatial_ft'] + \
                            combine_score[:,:,1].unsqueeze(-1)*ft['cap_ft'] 
                    if self.a_N>0:
                        ft['encoded_ft'] += combine_score[:,:,3].unsqueeze(-1)*ft['audio_ft']
            if self.v_N>0 and self.c_N==0 and self.args.enc_vc_combine == 'dyn':
                temp = torch.cat([ft['encoded_query']], dim=-1)
                if self.args.t2s:
                    temp = torch.cat([temp, ft['spatial_ft']], dim=-1)
                if self.args.s2t:
                    temp = torch.cat([temp, ft['temporal_ft']], dim=-1)
                if self.a_N>0:
                    temp = torch.cat([temp, ft['audio_ft']], dim=-1)
                combine_score = F.softmax((self.vc_combine_W(temp)), dim=-1)
                if self.args.t2s and self.args.s2t:
                    ft['encoded_ft'] = combine_score[:,:,0].unsqueeze(-1)*ft['temporal_ft'] + \
                        combine_score[:,:,1].unsqueeze(-1)*ft['spatial_ft']               
                if self.a_N>0:
                    ft['encoded_ft'] += combine_score[:,:,2].unsqueeze(-1)*ft['audio_ft']
            x = layer(b, ft, x)
            count+=1
            
        ft['decoded_text'] = self.norm(x)
        return ft
