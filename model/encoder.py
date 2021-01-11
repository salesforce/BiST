import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable
from model.modules import *
import pdb 

class Encoder(nn.Module):
    def __init__(self, size, nb_layers):
        super(Encoder, self).__init__()
        self.norm = nn.ModuleList()
        self.nb_layers = nb_layers
        for n in range(nb_layers):
            self.norm.append(LayerNorm(size))

    def forward(self, *seqs):
        output = []
        i=0
        seq_i=0
        while True: 
            if isinstance(seqs[seq_i],list):
                output_seq = []
                for seq in seqs[seq_i]:
                    output_seq.append(self.norm[i](seq))
                    i+=1
                output.append(output_seq)
                seq_i+=1
            else:
                if seqs[seq_i] is None:
                    output.append(None)
                    seq_i+=1
                else:
                    output.append(self.norm[i](seqs[seq_i]))
                    i+=1
                    seq_i+=1
            if seq_i == len(seqs): 
                break
        return output 

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, ff1, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.ff1 = ff1
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size 

    def forward(self, seq, seq_mask):
        seq = self.sublayer[0](seq, lambda seq: self.self_attn(seq, seq, seq, seq_mask))
        return self.sublayer[1](seq, self.ff1)
    
class VidEncoder(nn.Module):
    def __init__(self, temporal_W, spatial_W, vid_position, layer, N):
        super(VidEncoder, self).__init__()
        self.N = N
        self.temporal_W = temporal_W
        self.spatial_W = spatial_W
        self.vid_position = vid_position
        self.layers = clones(layer, N)
        self.spatial2temporal_norm = LayerNorm(layer.size)
        self.temporal2spatial_norm = LayerNorm(layer.size)

    def forward(self, b, pooled_query, ft):
        # TODO: assuming only one features and it's spatiotemporal feature
        temporal2spatial_in = self.temporal_W(b.temporal_ft)
        spatial2temporal_in = self.spatial_W(b.spatial_ft)
        #temporal2spatial_in = vid_ft.mean(2)
        #spatial2temporal_in = vid_ft.mean(1)
        if hasattr(self, "vid_position") and self.vid_position is not None:
            temporal2spatial_in = self.vid_position(temporal2spatial_in)
            spatial2temporal_in = self.vid_position(spatial2temporal_in)
        in_ft = {}
        in_ft['temporal2spatial'] = temporal2spatial_in
        in_ft['spatial2temporal'] = spatial2temporal_in
        for layer in self.layers:
            in_ft = layer(in_ft, pooled_query, b)
        
        #ft['encoded_vid_t2s']  = self.temporal2spatial_norm(in_ft['spatial2temporal'])
        #ft['encoded_vid_s2t'] = self.spatial2temporal_norm(in_ft['temporal2spatial'])
        
        ft['spatial_ft']  = self.temporal2spatial_norm(in_ft['spatial2temporal'])
        ft['temporal_ft'] = self.spatial2temporal_norm(in_ft['temporal2spatial'])
        
        return ft
            
class VidEncoderLayer(nn.Module):
    def __init__(self, size, attn, nb_attn, temporal2spatial_ff, spatial2temporal_ff, dropout, args):
        super(VidEncoderLayer, self).__init__()
        self.size = size
        self.temporal2spatial_attn = clones(attn, nb_attn)
        self.spatial2temporal_attn = clones(attn, nb_attn)
        self.temporal2spatial_ff = temporal2spatial_ff
        self.spatial2temporal_ff = spatial2temporal_ff
        self.sublayer = clones(SublayerConnection(size, dropout), (nb_attn+1)*2)
        self.args = args 

    def forward(self, in_ft, pooled_query, b):
        t2s = in_ft['temporal2spatial']
        s2t = in_ft['spatial2temporal']
        # TODO: assuming only one features and it's spatiotemporal feature
        s_t2s = self.sublayer[0](t2s, lambda t2s: self.temporal2spatial_attn[0](t2s, t2s, t2s, b.temporal_mask[0]))
        s_s2t = self.sublayer[1](s2t, lambda s2t: self.spatial2temporal_attn[0](s2t, s2t, s2t, b.spatial_mask[0]))
        
        q_t2s = self.sublayer[2](pooled_query, lambda pooled_query: self.temporal2spatial_attn[1](pooled_query, s_t2s, s_t2s, b.temporal_mask[0]))
        q_s2t = self.sublayer[3](pooled_query, lambda pooled_query: self.spatial2temporal_attn[1](pooled_query, s_s2t, s_s2t, b.spatial_mask[0]))
                               
        if self.args.enc_st_combine == 'sum':
            proj_t2s = q_t2s + s_s2t
            proj_s2t = q_s2t + s_t2s
        else:
            proj_t2s = q_t2s * s_s2t
            proj_s2t = q_s2t * s_t2s
        
        out_t2s = self.sublayer[4](proj_t2s, lambda proj_t2s: self.temporal2spatial_attn[2](proj_t2s, proj_t2s, proj_t2s, b.spatial_mask[0]))
        out_s2t = self.sublayer[5](proj_s2t, lambda proj_s2t: self.spatial2temporal_attn[2](proj_s2t, proj_s2t, proj_s2t, b.temporal_mask[0]))
        
        in_ft['spatial2temporal'] = self.sublayer[6](out_t2s, self.temporal2spatial_ff)
        in_ft['temporal2spatial'] = self.sublayer[7](out_s2t, self.spatial2temporal_ff)
        
        return in_ft 
    
class VidEncoder2(nn.Module):
    def __init__(self, temporal_W, spatial_W, vid_position, layer, N):
        super(VidEncoder2, self).__init__()
        self.N = N
        self.temporal_W = temporal_W
        self.spatial_W = spatial_W
        self.vid_position = vid_position
        self.layers = clones(layer, N)
        self.spatial_in_norm = LayerNorm(layer.size)
        self.temporal_in_norm = LayerNorm(layer.size)
        self.spatial_out_norm = LayerNorm(layer.size)
        self.temporal_out_norm = LayerNorm(layer.size)

    def forward(self, b, query, query_mask, ft):
        # TODO: assuming only one features and it's spatiotemporal feature
        temporal_ft = self.temporal_W(b.temporal_ft)
        spatial_ft = self.spatial_W(b.spatial_ft)
        if hasattr(self, "vid_position") and self.vid_position is not None:
            temporal_ft = self.vid_position(temporal_ft)
            spatial_ft = self.vid_position(spatial_ft)
        temporal_ft = self.temporal_in_norm(temporal_ft)
        spatial_ft = self.spatial_in_norm(spatial_ft)
        in_ft = {}
        in_ft['temporal'] = query
        in_ft['spatial'] = query
        for layer in self.layers:
            in_ft = layer(in_ft, temporal_ft, spatial_ft, b, query_mask)
        
        ft['temporal_ft']  = self.temporal_out_norm(in_ft['temporal'])
        ft['spatial_ft'] = self.spatial_out_norm(in_ft['spatial'])
        
        return ft
            
class VidEncoder2a(nn.Module):
    def __init__(self, temporal_W, spatial_W, vid_position, layer, N):
        super(VidEncoder2a, self).__init__()
        self.N = N
        self.temporal_W = temporal_W
        self.spatial_W = spatial_W
        self.vid_position = vid_position
        self.layers = clones(layer, N)
        self.spatial_in_norm = LayerNorm(layer.size)
        self.temporal_in_norm = LayerNorm(layer.size)
        self.spatial_out_norm = LayerNorm(layer.size)
        self.temporal_out_norm = LayerNorm(layer.size)

    def forward(self, b, query, query_mask, ft):
        # TODO: assuming only one features and it's spatiotemporal feature
        temporal_ft = nn.ReLU()(self.temporal_W(b.temporal_ft))
        spatial_ft = nn.ReLU()(self.spatial_W(b.spatial_ft))
        if hasattr(self, "vid_position") and self.vid_position is not None:
            temporal_ft = self.vid_position(temporal_ft)
            spatial_ft = self.vid_position(spatial_ft)
        temporal_ft = self.temporal_in_norm(temporal_ft)
        spatial_ft = self.spatial_in_norm(spatial_ft)
        in_ft = {}
        in_ft['temporal'] = query
        in_ft['spatial'] = query
        for layer in self.layers:
            in_ft = layer(in_ft, temporal_ft, spatial_ft, b, query_mask)
        
        ft['temporal_ft']  = self.temporal_out_norm(in_ft['temporal'])
        ft['spatial_ft'] = self.spatial_out_norm(in_ft['spatial'])
        
        return ft        
        
class VidEncoderLayer2(nn.Module):
    def __init__(self, size, attn, nb_attn, temporal_ff, spatial_ff, dropout):
        super(VidEncoderLayer2, self).__init__()
        self.size = size
        self.temporal_attn = clones(attn, nb_attn)
        self.spatial_attn = clones(attn, nb_attn)
        self.temporal_ff = temporal_ff
        self.spatial_ff = spatial_ff
        self.sublayer = clones(SublayerConnection(size, dropout), (nb_attn+1)*2)

    def forward(self, in_ft, temporal_ft, spatial_ft, b):
        t = in_ft['temporal']
        s = in_ft['spatial']
        # TODO: assuming only one features and it's spatiotemporal feature
        s_t = self.sublayer[0](t, lambda t: self.temporal_attn[0](t, t, t, b.query_mask))
        s_s = self.sublayer[1](s, lambda s: self.spatial_attn[0](s, s, s, b.query_mask))
        
        q_t = self.sublayer[2](s_t, lambda s_t: self.temporal_attn[1](s_t, temporal_ft, temporal_ft, b.temporal_mask))
        q_s = self.sublayer[3](s_s, lambda s_s: self.spatial_attn[1](s_s, spatial_ft, spatial_ft, b.spatial_mask))
                                 
        in_ft['temporal'] = self.sublayer[4](q_t, self.temporal_ff)
        in_ft['spatial'] = self.sublayer[5](q_s, self.spatial_ff)
        
        return in_ft        
    
class VidEncoderLayer3(nn.Module):
    def __init__(self, size, attn, nb_attn, temporal_ff, spatial_ff, dropout):
        super(VidEncoderLayer3, self).__init__()
        self.size = size
        self.temporal_attn = clones(attn, nb_attn)
        self.spatial_attn = clones(attn, nb_attn)
        self.temporal_ff = temporal_ff
        self.spatial_ff = spatial_ff
        self.sublayer = clones(SublayerConnection(size, dropout), (nb_attn+1)*2)

    def forward(self, in_ft, temporal_ft, spatial_ft, b, query_mask):
        t = in_ft['temporal']
        s = in_ft['spatial']
        # TODO: assuming only one features and it's spatiotemporal feature
        s_t = self.sublayer[0](t, lambda t: self.temporal_attn[0](t, t, t, query_mask))
        s_s = self.sublayer[1](s, lambda s: self.spatial_attn[0](s, s, s, query_mask))
        
        s_t += s_s
        s_s += s_t
        
        q_t = self.sublayer[2](s_t, lambda s_t: self.temporal_attn[1](s_t, temporal_ft, temporal_ft, b.temporal_mask))
        q_s = self.sublayer[3](s_s, lambda s_s: self.spatial_attn[1](s_s, spatial_ft, spatial_ft, b.spatial_mask))
                                 
        in_ft['temporal'] = self.sublayer[4](q_t, self.temporal_ff)
        in_ft['spatial'] = self.sublayer[5](q_s, self.spatial_ff)
        
        return in_ft 
    
class VidEncoder3(nn.Module):
    def __init__(self, size, temporal_W, spatial_W, vid_position):
        super(VidEncoder3, self).__init__()
        self.temporal_W = temporal_W
        self.spatial_W = spatial_W
        self.vid_position = vid_position
        self.spatial_in_norm = LayerNorm(size)
        self.temporal_in_norm = LayerNorm(size)

    def forward(self, b, ft):
        temporal_ft = self.temporal_W(b.temporal_ft)
        spatial_ft = self.spatial_W(b.spatial_ft)
        if hasattr(self, "vid_position") and self.vid_position is not None:
            temporal_ft = self.vid_position(temporal_ft)
            spatial_ft = self.vid_position(spatial_ft)
        ft['temporal_ft'] = self.temporal_in_norm(temporal_ft)
        ft['spatial_ft'] = self.spatial_in_norm(spatial_ft)
        return ft

class VidEncoder4(nn.Module):
    def __init__(self, size, W, vid_position):
        super(VidEncoder4, self).__init__()
        self.W = W
        self.vid_position = vid_position
        self.norm = LayerNorm(size)

    def forward(self, b, ft):
        fts = self.W(b.fts)
        if self.vid_position is not None:
            pdb.set_trace()
            temporal_ft = self.vid_position(temporal_ft)
            spatial_ft = self.vid_position(spatial_ft)
        ft['spatiotemporal_ft'] = self.norm(fts)
        return ft
    
class VidEncoder5(nn.Module):
    def __init__(self, W, vid_position, layer, N):
        super(VidEncoder5, self).__init__()
        self.W = W
        self.N = N
        self.vid_position = vid_position
        self.layers = clones(layer, N)
        self.in_norm = LayerNorm(layer.size)
        self.spatial_out_norm = LayerNorm(layer.size)
        self.temporal_out_norm = LayerNorm(layer.size)

    def forward(self, b, ft):
        fts = self.W(b.fts)
        if self.vid_position is not None:
            temporal_ft = self.vid_position(temporal_ft)
            spatial_ft = self.vid_position(spatial_ft)
        fts = self.in_norm(fts)
        ft['spatiotemporal_ft'] = fts 
        in_ft = {}
        in_ft['t2s'] = ft['encoded_query']
        in_ft['s2t'] = ft['encoded_query']
        for layer in self.layers:
            in_ft = layer(in_ft, ft, b)
        ft['temporal_ft']  = self.temporal_out_norm(in_ft['s2t'])
        ft['spatial_ft'] = self.spatial_out_norm(in_ft['t2s'])   
        return ft
    
class VidEncoderLayer4(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, nb_ff, dropout, args):
        super(VidEncoderLayer4, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = clones(ff, nb_ff) 
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn + nb_ff)
        self.args = args
        if self.args.enc_st_combine in ['early_sum', 'early_dyn']:
            self.out_norm = LayerNorm(size)
            if self.args.enc_st_combine == 'early_dyn':
                self.st_combine_W = nn.Linear(size*3, 1)
            

    def temporal2spatial(self, b, ft, in_tensor):
        vft = ft['spatiotemporal_ft']
        perm_vft = vft.permute(0, 2, 1, 3)
        perm_vft = perm_vft.reshape(perm_vft.shape[0]*perm_vft.shape[1], perm_vft.shape[2], perm_vft.shape[3])
        
        exp_in = in_tensor.unsqueeze(1).expand(in_tensor.shape[0], vft.shape[2], in_tensor.shape[1], in_tensor.shape[2])
        exp_in = exp_in.reshape(exp_in.shape[0]*exp_in.shape[1], exp_in.shape[2], exp_in.shape[3])
        
        exp_temporal_mask = b.temporal_mask.unsqueeze(1).expand(
            b.temporal_mask.shape[0], vft.shape[2], b.temporal_mask.shape[1], b.temporal_mask.shape[2])
        exp_temporal_mask = exp_temporal_mask.reshape(exp_temporal_mask.shape[0]*exp_temporal_mask.shape[1], exp_temporal_mask.shape[2], exp_temporal_mask.shape[3])
        
        t_out = self.sublayer[self.sublayer_idx](exp_in, lambda exp_in: self.attn[self.attn_idx](exp_in, perm_vft, perm_vft, exp_temporal_mask)) 
        self.attn_idx += 1
        self.sublayer_idx +=1
        
        permuted_t_out = t_out.reshape(in_tensor.shape[0], -1, t_out.shape[1], t_out.shape[2]).permute(0, 2, 1, 3)
        permuted_t_out = permuted_t_out.reshape(permuted_t_out.shape[0]*permuted_t_out.shape[1], -1, permuted_t_out.shape[3])
        
        unsq_in = in_tensor.unsqueeze(2).reshape(in_tensor.shape[0]*in_tensor.shape[1], 1, in_tensor.shape[2])
        
        ts_out = self.sublayer[self.sublayer_idx](unsq_in, lambda unsq_in: self.attn[self.attn_idx](unsq_in, permuted_t_out, permuted_t_out, None)) 
        self.attn_idx +=1
        self.sublayer_idx += 1
        
        ts_out = ts_out.reshape(in_tensor.shape[0], -1, 1, in_tensor.shape[-1]).squeeze(2)
        ts_out = self.sublayer[self.sublayer_idx](ts_out, self.ff[self.ff_idx])
        self.sublayer_idx += 1
        self.ff_idx += 1
        
        return ts_out
    
    def spatial2temporal(self, b, ft, in_tensor):
        vft = ft['spatiotemporal_ft']
        rs_vft = vft.reshape(vft.shape[0]*vft.shape[1], vft.shape[2], vft.shape[3])
        
        exp_in = in_tensor.unsqueeze(1).expand(in_tensor.shape[0], vft.shape[1], in_tensor.shape[1], in_tensor.shape[2])
        exp_in = exp_in.reshape(exp_in.shape[0]*exp_in.shape[1], exp_in.shape[2], exp_in.shape[3])
        
        s_out = self.sublayer[self.sublayer_idx](exp_in, lambda exp_in: self.attn[self.attn_idx](exp_in, rs_vft, rs_vft, None))
        self.attn_idx += 1
        self.sublayer_idx +=1
        
        permuted_s_out = s_out.reshape(in_tensor.shape[0], -1, s_out.shape[1], s_out.shape[2]).permute(0, 2, 1, 3)
        permuted_s_out = permuted_s_out.reshape(permuted_s_out.shape[0]*permuted_s_out.shape[1], -1, permuted_s_out.shape[-1])
        
        unsq_in = in_tensor.unsqueeze(2).reshape(in_tensor.shape[0]*in_tensor.shape[1], 1, in_tensor.shape[2])
        
        exp_temporal_mask = b.temporal_mask.unsqueeze(1).expand(
            b.temporal_mask.shape[0], in_tensor.shape[1], b.temporal_mask.shape[1], b.temporal_mask.shape[2])
        exp_temporal_mask = exp_temporal_mask.reshape(exp_temporal_mask.shape[0]*exp_temporal_mask.shape[1], exp_temporal_mask.shape[2], exp_temporal_mask.shape[3])
        
        st_out = self.sublayer[self.sublayer_idx](unsq_in, lambda unsq_in: self.attn[self.attn_idx](unsq_in, permuted_s_out, permuted_s_out, exp_temporal_mask)) 
        self.attn_idx +=1
        self.sublayer_idx += 1
        
        st_out = st_out.reshape(in_tensor.shape[0], -1, 1, in_tensor.shape[-1]).squeeze(2)
        st_out = self.sublayer[self.sublayer_idx](st_out, self.ff[self.ff_idx])
        self.sublayer_idx += 1
        self.ff_idx += 1
        
        return st_out 
    
    def forward(self, in_ft, ft, b):
        self.attn_idx, self.ff_idx, self.sublayer_idx = 0, 0, 0 
        if not hasattr(self.args, 't2s') or self.args.t2s:
            t2s = in_ft['t2s']
            t2s = self.sublayer[self.sublayer_idx](t2s, lambda t2s: self.attn[self.attn_idx](t2s, t2s, t2s, b.query_mask))
            self.attn_idx +=1
            self.sublayer_idx += 1
            t2s = self.temporal2spatial(b, ft, t2s)
            in_ft['t2s'] = t2s
        
        if not hasattr(self.args, 's2t') or self.args.s2t:
            s2t = in_ft['s2t']
            s2t = self.sublayer[self.sublayer_idx](s2t, lambda s2t: self.attn[self.attn_idx](s2t, s2t, s2t, b.query_mask))
            self.attn_idx +=1
            self.sublayer_idx += 1
            s2t = self.spatial2temporal(b, ft, s2t) 
            in_ft['s2t'] = s2t
            
        if self.args.enc_st_combine == 'early_sum':
            temp = self.out_norm(t2s + s2t) 
            in_ft['s2t'] = temp
            in_ft['t2s'] = temp
        elif self.args.enc_st_combine == 'early_dyn':
            vec = torch.cat([ft['encoded_query'], t2s, s2t], dim=-1)
            scores = nn.Sigmoid()(self.st_combine_W(vec))
            temp = self.out_norm(scores*t2s + (1-scores)*s2t)
            in_ft['s2t'] = temp
            in_ft['t2s'] = temp
            
        return in_ft 
    
class VidEncoder5a(nn.Module):
    def __init__(self, W, vid_position, layer, N):
        super(VidEncoder5a, self).__init__()
        self.W = W
        self.N = N
        self.vid_position = vid_position
        self.layers = clones(layer, N)
        self.in_norm = LayerNorm(layer.size)
        self.spatial_out_norm = LayerNorm(layer.size)
        self.temporal_out_norm = LayerNorm(layer.size)

    def forward(self, b, ft):
        fts = nn.ReLU()(self.W(b.fts))
        if self.vid_position is not None:
            temp = fts.view(fts.shape[0], -1, fts.shape[-1])
            temp = self.vid_position(temp)
            fts = temp.view_as(fts)
        fts = self.in_norm(fts)
        ft['spatiotemporal_ft'] = fts 
        in_ft = {}
        in_ft['t2s'] = ft['encoded_query']
        in_ft['s2t'] = ft['encoded_query']
        for layer in self.layers:
            in_ft = layer(in_ft, ft, b)
        ft['temporal_ft']  = self.temporal_out_norm(in_ft['s2t'])
        ft['spatial_ft'] = self.spatial_out_norm(in_ft['t2s'])   
        return ft   
    
class VidEncoder6(nn.Module):
    def __init__(self, W, vid_position, v_layer, c_layer, v_N, c_N, args, a_W=None, a_layer=None, a_N=0):
        super(VidEncoder6, self).__init__()
        self.v_N = v_N
        self.c_N = c_N
        self.a_N = a_N
        self.args = args 
        
        if self.v_N>0:
            self.W = W
            self.vid_position = vid_position
            self.v_layers = clones(v_layer, v_N)
            self.in_norm = LayerNorm(v_layer.size)
            if self.args.enc_st_combine == 'none':
                self.spatial_out_norm = LayerNorm(v_layer.size)
                self.temporal_out_norm = LayerNorm(v_layer.size)
            elif self.args.enc_st_combine not in ['early_sum', 'early_dyn']:
                self.out_norm = LayerNorm(v_layer.size)
                if self.args.enc_st_combine == 'dyn':
                    self.st_combine_W = nn.Linear(v_layer.size*3, 1)
        
        if self.c_N>0:
            self.c_layers = clones(c_layer, c_N) 
            self.cap_out_norm = LayerNorm(c_layer.size)
            
        if self.a_N>0:
            self.a_W = a_W
            self.vid_position = vid_position
            self.a_layers = clones(a_layer, a_N) 
            self.a_in_norm = LayerNorm(a_layer.size)
            self.a_out_norm = LayerNorm(a_layer.size)
            
        if self.v_N>0 and self.c_N>0 and (hasattr(self.args, 'enc_vc_combine') and self.args.enc_vc_combine=='dyn'):
            if self.args.enc_st_combine in ['sum', 'dyn'] and self.args.s2t and self.args.t2s:
                self.vc_combine_W = nn.Linear(v_layer.size*3, 1)                
            else:
                self.vc_combine_W = nn.Linear(v_layer.size*4, 3)
                
    def forward(self, b, ft):
        in_ft = {}  
        if self.v_N>0:
            fts = nn.ReLU()(self.W(b.fts))
            if self.vid_position is not None:
                temp = fts.view(fts.shape[0], -1, fts.shape[-1])
                temp = self.vid_position(temp)
                fts = temp.view_as(fts)
            fts = self.in_norm(fts)
            ft['spatiotemporal_ft'] = fts 
            in_ft['t2s'] = ft['encoded_query']
            in_ft['s2t'] = ft['encoded_query']
            for v_layer in self.v_layers:
                in_ft = v_layer(in_ft, ft, b)
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
            in_ft['cap'] = ft['encoded_query']
            for c_layer in self.c_layers:
                in_ft = c_layer(in_ft, ft, b)          
            ft['cap_ft'] = self.cap_out_norm(in_ft['cap'])  
            
        if hasattr(self, "a_N") and self.a_N>0:
            audio_fts = nn.ReLU()(self.a_W(b.audio_fts))
            if self.vid_position is not None:
                audio_fts = self.vid_position(audio_fts)
            audio_fts = self.a_in_norm(audio_fts)
            ft['encoded_audio'] = audio_fts
            in_ft['audio'] = ft['encoded_query']
            for a_layer in self.a_layers:
                in_ft = a_layer(in_ft, ft, b)          
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
                temp = torch.cat([ft['encoded_query'], ft['temporal_ft'], ft['spatial_ft'], ft['cap_ft']], dim=-1)
                combine_score = F.softmax((self.vc_combine_W(temp)), dim=-1)
                ft['encoded_ft'] = combine_score[:,:,0].unsqueeze(-1)*ft['temporal_ft'] + \
                    combine_score[:,:,1].unsqueeze(-1)*ft['spatial_ft'] + \
                    combine_score[:,:,2].unsqueeze(-1)*ft['cap_ft'] 
        
        return ft   
    
class CapEncoderLayer(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, dropout):
        super(CapEncoderLayer, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = ff
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn + 1)
    
    def forward(self, in_ft, ft, b):
        c = in_ft['cap']
        # TODO: assuming only one features and it's spatiotemporal feature
        c = self.sublayer[0](c, lambda c: self.attn[0](c, c, c, b.query_mask))
        c = self.sublayer[1](c, lambda c: self.attn[1](c, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask)) 
        c = self.sublayer[2](c, self.ff)     
        in_ft['cap'] = c
        return in_ft

class VidEncoder7(nn.Module):
    def __init__(self, W, vid_position, v_layer, c_layer, v_N, c_N, args):
        super(VidEncoder7, self).__init__()
        self.v_N = v_N
        self.c_N = c_N
        self.args = args 
        
        if self.v_N>0:
            self.W = W
            self.vid_position = vid_position
            self.v_layers = clones(v_layer, v_N)
            self.in_norm = LayerNorm(v_layer.size)
            self.out_norm = LayerNorm(v_layer.size)
        
        if self.c_N>0:
            self.c_layers = clones(c_layer, c_N) 
            self.cap_out_norm = LayerNorm(c_layer.size)
        
    def forward(self, b, ft):
        in_ft = {}  
        if self.v_N>0:
            fts = nn.ReLU()(self.W(b.fts))
            if self.vid_position is not None:
                fts = self.vid_position(fts)
            fts = self.in_norm(fts)
            ft['spatiotemporal_ft'] = fts 
            in_ft['spatiotemporal_ft'] = ft['encoded_query']
            for v_layer in self.v_layers:
                in_ft = v_layer(in_ft, ft, b)
            # dummy name: temporal_ft
            ft['temporal_ft'] = self.out_norm(in_ft['spatiotemporal_ft'])
        if self.c_N>0:
            in_ft['cap'] = ft['encoded_query']
            for c_layer in self.c_layers:
                in_ft = c_layer(in_ft, ft, b)          
            ft['cap_ft'] = self.cap_out_norm(in_ft['cap'])  
        return ft   
    
class CapEncoderLayer(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, dropout):
        super(CapEncoderLayer, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = ff
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn + 1)
    
    def forward(self, in_ft, ft, b):
        c = in_ft['cap']
        # TODO: assuming only one features and it's spatiotemporal feature
        c = self.sublayer[0](c, lambda c: self.attn[0](c, c, c, b.query_mask))
        c = self.sublayer[1](c, lambda c: self.attn[1](c, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask)) 
        c = self.sublayer[2](c, self.ff)     
        in_ft['cap'] = c
        return in_ft
    
class AudioEncoderLayer(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, dropout):
        super(AudioEncoderLayer, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = ff
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn + 1)
    
    def forward(self, in_ft, ft, b):
        a = in_ft['audio']
        # TODO: assuming only one features and it's spatiotemporal feature
        a = self.sublayer[0](a, lambda a: self.attn[0](a, a, a, b.query_mask))
        a = self.sublayer[1](a, lambda a: self.attn[1](a, ft['encoded_audio'], ft['encoded_audio'], b.audio_mask)) 
        a = self.sublayer[2](a, self.ff)     
        in_ft['audio'] = a
        return in_ft
    
class VidEncoderLayer5(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, dropout):
        super(VidEncoderLayer5, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = ff
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn + 1)
    
    def forward(self, in_ft, ft, b):
        vft = in_ft['spatiotemporal_ft']
        # TODO: assuming only one features and it's spatiotemporal feature
        vft = self.sublayer[0](vft, lambda vft: self.attn[0](vft, vft, vft, b.query_mask))
        vft = self.sublayer[1](vft, lambda vft: self.attn[1](vft, ft['spatiotemporal_ft'], ft['spatiotemporal_ft'], b.fts_mask)) 
        vft = self.sublayer[2](vft, self.ff)     
        in_ft['spatiotemporal_ft'] = vft
        return in_ft
    
class VidEncoder8(nn.Module):
    def __init__(self, W, a_W, vid_position, v_N, a_N, size, args):
        super(VidEncoder8, self).__init__()
        self.v_N = v_N
        self.a_N = a_N
        self.args = args 
        
        if self.v_N>0:
            self.W = W
            self.vid_position = vid_position
            self.in_norm = LayerNorm(size)
            
        if self.a_N>0:
            self.a_W = a_W
            self.vid_position = vid_position
            self.a_in_norm = LayerNorm(size)
            
    def forward(self, b, ft):
        in_ft = {}  
        if self.v_N>0:
            fts = nn.ReLU()(self.W(b.fts))
            if self.vid_position is not None:
                temp = fts.view(fts.shape[0], -1, fts.shape[-1])
                temp = self.vid_position(temp)
                fts = temp.view_as(fts)
            fts = self.in_norm(fts)
            ft['spatiotemporal_ft'] = fts  
            
        if self.a_N>0:
            if self.args.noW_venc:
                audio_fts = b.audio_fts
            else:
                audio_fts = nn.ReLU()(self.a_W(b.audio_fts))
            if self.vid_position is not None:
                audio_fts = self.vid_position(audio_fts)
            audio_fts = self.a_in_norm(audio_fts)
            ft['encoded_audio'] = audio_fts
            
        return ft   
    
