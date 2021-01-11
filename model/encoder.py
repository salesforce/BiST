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
    
