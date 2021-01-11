import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable
import pdb
from model.modules import *

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, b, ft, x):
        for layer in self.layers:
            x = layer(b, ft, x)
        ft['decoded_text'] = self.norm(x)
        return ft

class DecoderLayer(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = ff
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn+1)

    def forward(self, b, ft, x):
        x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        x = self.sublayer[1](x, lambda x: self.attn[1](x, ft['encoded_his'], ft['encoded_his'], b.his_mask))
        x = self.sublayer[2](x, lambda x: self.attn[2](x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
        x = self.sublayer[3](x, lambda x: self.attn[3](x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
        return self.sublayer[4](x, self.ff)
    
class MultimodalDecoder(nn.Module):
    def __init__(self, layer, N, dec_st_combine):
        super(MultimodalDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.dec_st_combine = dec_st_combine

    def forward(self, b, ft):
        in_ft = {}
        in_ft['temporal2spatial'] = ft['decoded_text']
        in_ft['spatial2temporal'] = ft['decoded_text']
        for layer in self.layers:
            in_ft = layer(b, ft, in_ft)
        if hasattr(self, 'dec_st_combine') and self.dec_st_combine == 'sum':
            combined = in_ft['temporal2spatial'] + in_ft['spatial2temporal']
        else:
            combined = in_ft['temporal2spatial'] * in_ft['spatial2temporal']
        ft['decoded_text'] = self.norm(combined)
        return ft

class MultimodalDecoderLayer(nn.Module):
    def __init__(self, size, attn, nb_attn, temporal2spatial_ff, spatial2temporal_ff, dropout):
        super(MultimodalDecoderLayer, self).__init__()
        self.size = size
        self.temporal2spatial_attn = clones(attn, nb_attn)
        self.spatial2temporal_attn = clones(attn, nb_attn)
        self.temporal2spatial_ff = temporal2spatial_ff
        self.spatial2temporal_ff = spatial2temporal_ff
        self.sublayer = clones(SublayerConnection(size, dropout), (nb_attn+1)*2)

    def forward(self, b, ft, in_ft):
        t2s = in_ft['temporal2spatial']
        s2t = in_ft['spatial2temporal']
        temporal = ft['encoded_vid_s2t']
        spatial = ft['encoded_vid_t2s']
        
        t2s = self.sublayer[0](t2s, lambda t2s: self.temporal2spatial_attn[0](t2s, temporal, temporal, b.temporal_mask[0]))
        s2t = self.sublayer[1](s2t, lambda s2t: self.spatial2temporal_attn[0](s2t, spatial, spatial, b.spatial_mask[0]))
        
        t2s = self.sublayer[2](t2s, lambda t2s: self.temporal2spatial_attn[1](t2s, spatial, spatial, b.spatial_mask[0]))
        s2t = self.sublayer[3](s2t, lambda s2t: self.spatial2temporal_attn[1](s2t, temporal, temporal, b.temporal_mask[0]))

        in_ft['temporal2spatial'] = self.sublayer[4](t2s, self.temporal2spatial_ff)
        in_ft['spatial2temporal'] = self.sublayer[5](s2t, self.spatial2temporal_ff)
        
        return in_ft
    
class MultimodalDecoder2(nn.Module):
    def __init__(self, layer, N, dec_st_combine):
        super(MultimodalDecoder2, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.dec_st_combine = dec_st_combine

    def forward(self, b, ft):
        in_ft = {}
        in_ft['temporal'] = ft['decoded_text']
        in_ft['spatial'] = ft['decoded_text']
        for layer in self.layers:
            in_ft = layer(b, ft, in_ft)
        if hasattr(self, 'dec_st_combine') and self.dec_st_combine == 'sum':
            combined = in_ft['temporal'] + in_ft['spatial']
        else:
            combined = in_ft['temporal'] * in_ft['spatial']
        ft['decoded_text'] = self.norm(combined)
        return ft

class MultimodalDecoderLayer2(nn.Module):
    def __init__(self, size, attn, nb_attn, temporal_ff, spatial_ff, dropout):
        super(MultimodalDecoderLayer2, self).__init__()
        self.size = size
        self.temporal_attn = clones(attn, nb_attn)
        self.spatial_attn = clones(attn, nb_attn)
        self.temporal_ff = temporal_ff
        self.spatial_ff = spatial_ff
        self.sublayer = clones(SublayerConnection(size, dropout), (nb_attn+1)*2)

    def forward(self, b, ft, in_ft):
        t = in_ft['temporal']
        s = in_ft['spatial']
        temporal = ft['temporal_ft']
        spatial = ft['spatial_ft']
        
        t = self.sublayer[0](t, lambda t: self.temporal_attn[0](t, temporal, temporal, b.query_mask))       
        s = self.sublayer[1](s, lambda s: self.spatial_attn[0](s, spatial, spatial, b.query_mask))

        in_ft['temporal'] = self.sublayer[2](t, self.temporal_ff)
        in_ft['spatial'] = self.sublayer[3](s, self.spatial_ff)
        
        return in_ft
    
class MultimodalDecoder3(nn.Module):
    def __init__(self, layer, N):
        super(MultimodalDecoder3, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, b, ft, x):
        for layer in self.layers:
            x = layer(b, ft, x)
        ft['decoded_text'] = self.norm(x)
        return ft

class MultimodalDecoderLayer3(nn.Module):
    def __init__(self, size, attn, nb_attn, v_attn, nb_v_attn, temporal_ff, spatial_ff, dropout, args):
        super(MultimodalDecoderLayer3, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.temporal_attn = clones(v_attn, nb_v_attn)
        self.spatial_attn = clones(v_attn, nb_v_attn)
        self.temporal_ff = temporal_ff
        self.spatial_ff = spatial_ff
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn+(nb_v_attn+1)*2)
        self.args = args 

    def forward(self, b, ft, x):
        x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        x = self.sublayer[1](x, lambda x: self.attn[1](x, ft['encoded_his'], ft['encoded_his'], b.his_mask))
        x = self.sublayer[2](x, lambda x: self.attn[2](x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
        x = self.sublayer[3](x, lambda x: self.attn[3](x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
        
        temporal = ft['temporal_ft']
        spatial = ft['spatial_ft']
        
        if self.args.vid_enc_mode == 6:
            t = self.sublayer[4](x, lambda x: self.temporal_attn[0](x, temporal, temporal, b.temporal_mask))       
            s = self.sublayer[5](x, lambda x: self.spatial_attn[0](x, spatial, spatial, b.spatial_mask))
        else:
            t = self.sublayer[4](x, lambda x: self.temporal_attn[0](x, temporal, temporal, b.query_mask))       
            s = self.sublayer[5](x, lambda x: self.spatial_attn[0](x, spatial, spatial, b.query_mask))
        
        t = self.sublayer[6](t, self.temporal_ff)
        s = self.sublayer[7](s, self.spatial_ff)
        
        if hasattr(self, 'args') and self.args.dec_st_combine == 'sum':
            combined = t + s
        else:
            combined = t * s
        
        return combined

class MultimodalDecoder4(nn.Module):
    def __init__(self, layer, N):
        super(MultimodalDecoder4, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, b, ft, x):
        in_ft = {}
        in_ft['decoded_text'] = x
        in_ft['temporal'] = ft['encoded_query']
        in_ft['spatial'] = ft['encoded_query']
        for layer in self.layers:
            in_ft = layer(b, ft, in_ft)
        ft['decoded_text'] = self.norm(in_ft['decoded_text'])
        ft['temporal_out'] = self.norm(in_ft['temporal'])
        ft['spatial_out'] = self.norm(in_ft['spatial'])
        return ft
    
class MultimodalDecoderLayer4(nn.Module):
    def __init__(self, size, attn, nb_attn, v_attn, nb_v_attn, q_attn, nb_q_attn, 
                 temporal_ff, spatial_ff, q_temporal_ff, q_spatial_ff,
                 dropout, args):
        super(MultimodalDecoderLayer4, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.temporal_attn = clones(v_attn, nb_v_attn)
        self.spatial_attn = clones(v_attn, nb_v_attn)
        self.q_temporal_attn = clones(q_attn, nb_q_attn)
        self.q_spatial_attn = clones(q_attn, nb_q_attn)
        self.temporal_ff = temporal_ff
        self.spatial_ff = spatial_ff
        self.q_temporal_ff = q_temporal_ff
        self.q_spatial_ff = q_spatial_ff
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn+(nb_v_attn+1)*2 + (nb_q_attn+1)*2)
        self.args = args 

    def forward(self, b, ft, in_ft):
        x = in_ft['decoded_text']
        x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        x = self.sublayer[1](x, lambda x: self.attn[1](x, ft['encoded_his'], ft['encoded_his'], b.his_mask))
        x = self.sublayer[2](x, lambda x: self.attn[2](x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
        x = self.sublayer[3](x, lambda x: self.attn[3](x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
        
        t = in_ft['temporal']
        s = in_ft['spatial']
        s_t = self.sublayer[4](t, lambda t: self.q_temporal_attn[0](t, t, t, b.query_mask))
        s_s = self.sublayer[5](s, lambda s: self.q_spatial_attn[0](s, s, s, b.query_mask))
        
        s_t += s_s
        s_s += s_t
        
        q_t = self.sublayer[6](s_t, lambda s_t: self.q_temporal_attn[1](s_t, ft['temporal_ft'], ft['temporal_ft'], b.temporal_mask))
        q_s = self.sublayer[7](s_s, lambda s_s: self.q_spatial_attn[1](s_s, ft['spatial_ft'], ft['spatial_ft'], b.spatial_mask))
                                 
        in_ft['temporal'] = self.sublayer[8](q_t, self.q_temporal_ff)
        in_ft['spatial'] = self.sublayer[9](q_s, self.q_spatial_ff)
        
        t = self.sublayer[10](x, lambda x: self.temporal_attn[0](x, in_ft['temporal'], in_ft['temporal'], b.query_mask))       
        s = self.sublayer[11](x, lambda x: self.spatial_attn[0](x, in_ft['spatial'], in_ft['spatial'], b.query_mask))
        
        t = self.sublayer[12](t, self.temporal_ff)
        s = self.sublayer[13](s, self.spatial_ff)
        
        if hasattr(self, 'args') and self.args.dec_st_combine == 'sum':
            combined = t + s
        else:
            combined = t * s
            
        in_ft['decoded_text'] = combined
   
        return in_ft

class MultimodalDecoder5(nn.Module):
    def __init__(self, layer, N):
        super(MultimodalDecoder5, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, b, ft, x):
        for layer in self.layers:
            x = layer(b, ft, x)
        ft['decoded_text'] = self.norm(x)
        return ft
    
class MultimodalDecoderLayer5(nn.Module):
    def __init__(self, size, attn, nb_attn, v_attn, nb_v_attn, temporal_ff, spatial_ff, cap_ff, que_ff, ff, dropout, args):
        super(MultimodalDecoderLayer5, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.temporal_attn = clones(v_attn, nb_v_attn)
        self.spatial_attn = clones(v_attn, nb_v_attn)
        self.temporal_ff = clones(temporal_ff, nb_v_attn)
        self.spatial_ff = clones(spatial_ff, nb_v_attn)
        self.cap_ff = cap_ff
        self.que_ff = que_ff
        self.ff = ff
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn+nb_v_attn*4+3)
        self.args = args 
        if self.args.dynamic_fusion>0:
            if self.args.dynamic_fusion in [1,3]:
                fs_size = 5
            elif self.args.dynamic_fusion in [2,4,5,6]:
                fs_size = 3
            elif self.args.dynamic_fusion in [7]:
                fs_size = 2
            elif self.args.dynamic_fusion in [8]:
                fs_size = 1
            self.c_x_W = nn.Linear(size*fs_size, 1)
            self.q_x_W = nn.Linear(size*fs_size, 1)
            self.x_W = nn.Linear(size*fs_size, 1)
            if self.args.dynamic_fusion in [8]:
                self.c_x_w1 = nn.Linear(size, size)
                self.q_x_w1 = nn.Linear(size, size)
                self.c_x_w2 = nn.Linear(size, size)
                self.q_x_w2 = nn.Linear(size, size)
                self.x_w1 = nn.Linear(size, size)
                self.x_w2 = nn.Linear(size, size)
                

    def forward(self, b, ft, x):
        s_x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        h_x = self.sublayer[1](s_x, lambda s_x: self.attn[1](s_x, ft['encoded_his'], ft['encoded_his'], b.his_mask))
        
        c_x = self.sublayer[2](h_x, lambda h_x: self.attn[2](h_x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
        q_x = self.sublayer[3](h_x, lambda h_x: self.attn[3](h_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
        
        temporal = ft['temporal_ft']
        spatial = ft['spatial_ft']
        
        if self.args.vid_enc_mode in [13, 14, 17, 19]:
            if self.args.query_mm == 'query':
                temporal_mask = b.query_mask
                spatial_mask = b.query_mask
            elif self.args.query_mm == 'caption':
                temporal_mask = b.cap_mask
                spatial_mask = b.cap_mask
        else:
            temporal_mask = b.temporal_mask
            spatial_mask = b.spatial_mask
        
        t_c_x = self.sublayer[4](c_x, lambda c_x: self.temporal_attn[0](c_x, temporal, temporal, temporal_mask)) 
        t_c_x = self.sublayer[6](t_c_x, self.temporal_ff[0])

        s_c_x = self.sublayer[5](c_x, lambda c_x: self.spatial_attn[0](c_x, spatial, spatial, spatial_mask))
        s_c_x = self.sublayer[7](s_c_x, self.spatial_ff[0])

        t_q_x = self.sublayer[8](q_x, lambda q_x: self.temporal_attn[1](q_x, temporal, temporal, temporal_mask))       
        t_q_x = self.sublayer[10](t_q_x, self.temporal_ff[1])

        s_q_x = self.sublayer[9](q_x, lambda q_x: self.spatial_attn[1](q_x, spatial, spatial, spatial_mask))
        s_q_x = self.sublayer[11](s_q_x, self.spatial_ff[1])

        
        if self.args.dynamic_fusion>0:
            if self.args.dynamic_fusion in [1,3]:
                c_x_fs_vec = torch.cat([t_c_x.mean(1), s_c_x.mean(1), c_x.mean(1), temporal.mean(1),  spatial.mean(1)], dim=-1)
                q_x_fs_vec = torch.cat([t_q_x.mean(1), s_q_x.mean(1), q_x.mean(1), temporal.mean(1),  spatial.mean(1)], dim=-1)
            elif self.args.dynamic_fusion in [2,4]:
                c_x_fs_vec = torch.cat([t_c_x.mean(1), s_c_x.mean(1), c_x.mean(1)], dim=-1)
                q_x_fs_vec = torch.cat([t_q_x.mean(1), s_q_x.mean(1), q_x.mean(1)], dim=-1)
            elif self.args.dynamic_fusion in [5]:
                c_x_fs_vec = torch.cat([mask_mean(t_c_x, b.trg_mean_mask), 
                                        mask_mean(s_c_x, b.trg_mean_mask), 
                                        mask_mean(c_x, b.trg_mean_mask)], dim=-1)
                q_x_fs_vec = torch.cat([mask_mean(t_q_x, b.trg_mean_mask), 
                                        mask_mean(s_q_x, b.trg_mean_mask), 
                                        mask_mean(q_x, b.trg_mean_mask)], dim=-1)
            elif self.args.dynamic_fusion in [6]:
                c_x_fs_vec = torch.cat([t_c_x, s_c_x, c_x], dim=-1)
                q_x_fs_vec = torch.cat([t_q_x, s_q_x, q_x], dim=-1)
            elif self.args.dynamic_fusion in [7]:
                c_x_fs_vec = torch.cat([t_c_x, s_c_x], dim=-1)
                q_x_fs_vec = torch.cat([t_q_x, s_q_x], dim=-1)
            elif self.args.dynamic_fusion in [8]:
                c_x_fs_vec = F.relu(self.c_x_w1(t_c_x) + self.c_x_w2(s_c_x))
                q_x_fs_vec = F.relu(self.q_x_w1(t_q_x) + self.q_x_w2(s_q_x))
            #c_x_fs_wt = nn.Sigmoid()(self.c_x_W(c_x_fs_vec)).unsqueeze(-1)
            #q_x_fs_wt = nn.Sigmoid()(self.q_x_W(q_x_fs_vec)).unsqueeze(-1)
            c_x_fs_wt = nn.Sigmoid()(self.c_x_W(c_x_fs_vec))
            q_x_fs_wt = nn.Sigmoid()(self.q_x_W(q_x_fs_vec))
            ts_c = c_x_fs_wt*t_c_x + (1-c_x_fs_wt)*s_c_x
            ts_q = q_x_fs_wt*t_q_x + (1-q_x_fs_wt)*s_q_x
        else:
            if self.args.dec_st_combine == 'sum':
                ts_c = t_c_x + s_c_x
                ts_q = t_q_x + s_q_x
                if self.args.skip_connect:
                    ts_c += h_x
                    ts_q += h_x
            else:
                ts_c = t_c_x * s_c_x
                ts_q = t_q_x * s_q_x
                if self.args.skip_connect:
                    ts_c *= h_x
                    ts_q *= h_x
        
        if not self.args.no_ff:
            ts_c = self.sublayer[12](ts_c, self.cap_ff)
            ts_q = self.sublayer[13](ts_q, self.que_ff)
        
        if self.args.dynamic_fusion>0:
            if self.args.dynamic_fusion==1:
                x_fs_vec = torch.cat([c_x.mean(1), q_x.mean(1), h_x.mean(1), temporal.mean(1),  spatial.mean(1)], dim=-1)
            elif self.args.dynamic_fusion==2:
                x_fs_vec = torch.cat([c_x.mean(1), q_x.mean(1), h_x.mean(1)], dim=-1)
            if self.args.dynamic_fusion==3:
                x_fs_vec = torch.cat([ts_c.mean(1), ts_q.mean(1), h_x.mean(1), temporal.mean(1),  spatial.mean(1)], dim=-1)
            elif self.args.dynamic_fusion==4:
                x_fs_vec = torch.cat([ts_c.mean(1), ts_q.mean(1), h_x.mean(1)], dim=-1)
            elif self.args.dynamic_fusion==5:
                x_fs_vec = torch.cat([mask_mean(ts_c, b.trg_mean_mask), 
                                        mask_mean(ts_q, b.trg_mean_mask), 
                                        mask_mean(h_x, b.trg_mean_mask)], dim=-1)
            elif self.args.dynamic_fusion in [6]:
                x_fs_vec = torch.cat([ts_c, ts_q, h_x], dim=-1)
            elif self.args.dynamic_fusion in [7]:
                x_fs_vec = torch.cat([ts_c, ts_q], dim=-1)
            elif self.args.dynamic_fusion in [8]:
                x_fs_vec = F.relu(self.x_w1(ts_c) + self.x_w2(ts_q))
            #x_fs_wt = nn.Sigmoid()(self.x_W(x_fs_vec)).unsqueeze(-1)
            x_fs_wt = nn.Sigmoid()(self.x_W(x_fs_vec))
            combined = x_fs_wt*ts_c + (1-x_fs_wt)*ts_q
        else:
            if self.args.dec_st_combine == 'sum':
                combined = ts_c + ts_q
                if self.args.skip_connect:
                    combined += s_x
            else:
                combined = ts_c * ts_q
                if self.args.skip_connect:
                    combined *= s_x
        
        if not self.args.no_ff:
            combined = self.sublayer[14](combined, self.ff)

        return combined

class MultimodalDecoderLayer5a(nn.Module):
    def __init__(self, size, attn, nb_attn, v_attn, nb_v_attn, temporal_ff, spatial_ff, cap_ff, que_ff, ff, dropout, args):
        super(MultimodalDecoderLayer5a, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.temporal_attn = clones(v_attn, nb_v_attn)
        self.spatial_attn = clones(v_attn, nb_v_attn)
        self.temporal_ff = clones(temporal_ff, nb_v_attn)
        self.spatial_ff = clones(spatial_ff, nb_v_attn)
        self.cap_ff = cap_ff
        self.que_ff = que_ff
        self.ff = ff
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn+nb_v_attn*2+1)
        self.ts_sublayer = clones(SublayerConnection(int(size/2), dropout), nb_v_attn*2+2)
        self.args = args 

    def forward(self, b, ft, x):
        s_x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        h_x = self.sublayer[1](s_x, lambda s_x: self.attn[1](s_x, ft['encoded_his'], ft['encoded_his'], b.his_mask))
        
        c_x = self.sublayer[2](h_x, lambda h_x: self.attn[2](h_x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
        q_x = self.sublayer[3](h_x, lambda h_x: self.attn[3](h_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
        
        temporal = ft['temporal_ft']
        spatial = ft['spatial_ft']
        
        if self.args.vid_enc_mode in [13, 14, 17, 18]:
            if self.args.query_mm == 'query':
                temporal_mask = b.query_mask
                spatial_mask = b.query_mask
            elif self.args.query_mm == 'caption':
                temporal_mask = b.cap_mask
                spatial_mask = b.cap_mask
        else:
            temporal_mask = b.temporal_mask
            spatial_mask = b.spatial_mask
        
        t_c_x = self.sublayer[4](c_x, lambda c_x: self.temporal_attn[0](c_x, temporal, temporal, temporal_mask)) 
        pdb.set_trace()
        t_c_x = self.ts_sublayer[0](t_c_x, self.temporal_ff[0])

        s_c_x = self.sublayer[5](c_x, lambda c_x: self.spatial_attn[0](c_x, spatial, spatial, spatial_mask))
        s_c_x = self.ts_sublayer[1](s_c_x, self.spatial_ff[0])

        t_q_x = self.sublayer[6](q_x, lambda q_x: self.temporal_attn[1](q_x, temporal, temporal, temporal_mask))       
        t_q_x = self.ts_sublayer[2](t_q_x, self.temporal_ff[1])

        s_q_x = self.sublayer[7](q_x, lambda q_x: self.spatial_attn[1](q_x, spatial, spatial, spatial_mask))
        s_q_x = self.ts_sublayer[3](s_q_x, self.spatial_ff[1])

        ts_c = torch.cat([t_c_x,s_c_x], dim=-1)
        ts_q = torch.cat([t_q_x,s_q_x], dim=-1)
        
        pdb.set_trace()

        ts_c = self.ts_sublayer[4](ts_c, self.cap_ff)
        ts_q = self.ts_sublayer[5](ts_q, self.que_ff)
        
        combined = torch.cat([ts_c, ts_q], dim=-1)
        combined = self.sublayer[8](combined, self.ff)

        return combined
    
class MultimodalDecoderLayer6(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, nb_ff, nb_sublayers, dropout, args):
        super(MultimodalDecoderLayer6, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = clones(ff, nb_ff)
        self.sublayer = clones(SublayerConnection(size, dropout), nb_sublayers)
        self.args = args 
        if self.args.dynamic_fusion>0:
            if self.args.dynamic_fusion==1:
                fs_size = 5
            elif self.args.dynamic_fusion in [2,6]:
                fs_size = 3
            self.c_x_W = nn.Linear(size*fs_size, 1)
            self.q_x_W = nn.Linear(size*fs_size, 1)
            self.x_W = nn.Linear(size*fs_size, 1)
            
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
    
    def forward(self, b, ft, x):
        s_x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        h_x = self.sublayer[1](s_x, lambda s_x: self.attn[1](s_x, ft['encoded_his'], ft['encoded_his'], b.his_mask))
        
        c_x = self.sublayer[2](h_x, lambda h_x: self.attn[2](h_x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
        q_x = self.sublayer[3](h_x, lambda h_x: self.attn[3](h_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
        
        self.attn_idx, self.ff_idx, self.sublayer_idx = 4, 0, 4        
        ts_c_x = self.temporal2spatial(b, ft, c_x)        
        ts_q_x = self.temporal2spatial(b, ft, q_x)
        st_c_x = self.spatial2temporal(b, ft, c_x)
        st_q_x = self.spatial2temporal(b, ft, q_x)
                
        if self.args.dynamic_fusion>0:
            if self.args.dynamic_fusion==1:
                c_x_fs_vec = torch.cat([ts_c_x.mean(1), st_c_x.mean(1), c_x.mean(1), temporal.mean(1),  spatial.mean(1)], dim=-1)
                q_x_fs_vec = torch.cat([ts_q_x.mean(1), st_q_x.mean(1), q_x.mean(1), temporal.mean(1),  spatial.mean(1)], dim=-1)
            elif self.args.dynamic_fusion==2:
                c_x_fs_vec = torch.cat([ts_c_x.mean(1), st_c_x.mean(1), c_x.mean(1)], dim=-1)
                q_x_fs_vec = torch.cat([ts_q_x.mean(1), st_q_x.mean(1), q_x.mean(1)], dim=-1)
            elif self.args.dynamic_fusion in [6]:
                c_x_fs_vec = torch.cat([ts_c_x, st_c_x, c_x], dim=-1)
                q_x_fs_vec = torch.cat([ts_q_x, st_q_x, q_x], dim=-1)
            c_x_fs_wt = nn.Sigmoid()(self.c_x_W(c_x_fs_vec)) #.unsqueeze(-1)
            q_x_fs_wt = nn.Sigmoid()(self.q_x_W(q_x_fs_vec)) #.unsqueeze(-1)
            ts_c = c_x_fs_wt*ts_c_x + (1-c_x_fs_wt)*st_c_x
            ts_q = q_x_fs_wt*ts_q_x + (1-q_x_fs_wt)*st_q_x
        else:
            if self.args.dec_st_combine == 'sum':
                ts_c = ts_c_x + st_c_x
                ts_q = ts_q_x + st_q_x
                if self.args.skip_connect:
                    ts_c += h_x
                    ts_q += h_x
            else:
                ts_c = ts_c_x * st_c_x
                ts_q = ts_q_x * st_q_x
                if self.args.skip_connect:
                    ts_c *= h_x
                    ts_q *= h_x
        
        if not self.args.no_ff:
            ts_c = self.sublayer[self.sublayer_idx](ts_c, self.ff[self.ff_idx])
            self.ff_idx += 1
            self.sublayer_idx += 1
            ts_q = self.sublayer[self.sublayer_idx](ts_q, self.ff[self.ff_idx])
            self.ff_idx += 1
            self.sublayer_idx += 1
        
        if self.args.dynamic_fusion>0:
            if self.args.dynamic_fusion==1:
                x_fs_vec = torch.cat([ts_c.mean(1), ts_q.mean(1), h_x.mean(1), temporal.mean(1),  spatial.mean(1)], dim=-1)
            elif self.args.dynamic_fusion==2:
                x_fs_vec = torch.cat([ts_c.mean(1), ts_q.mean(1), h_x.mean(1)], dim=-1)
            elif self.args.dynamic_fusion in [6]:
                x_fs_vec = torch.cat([ts_c, ts_q, h_x], dim=-1)
            x_fs_wt = nn.Sigmoid()(self.x_W(x_fs_vec)) #.unsqueeze(-1)
            combined = x_fs_wt*ts_c + (1-x_fs_wt)*ts_q
        else:
            if self.args.dec_st_combine == 'sum':
                combined = ts_c + ts_q
                if self.args.skip_connect:
                    combined += s_x
            else:
                combined = ts_c * ts_q
                if self.args.skip_connect:
                    combined *= s_x
        
        if not self.args.no_ff:
            combined = self.sublayer[self.sublayer_idx](combined, self.ff[self.ff_idx])
            
        return combined
    
class MultimodalDecoderLayer7(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, nb_ff, nb_sublayers, dropout, args):
        super(MultimodalDecoderLayer7, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = clones(ff, nb_ff)
        self.sublayer = clones(SublayerConnection(size, dropout), nb_sublayers)
        self.args = args 

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
    
    def forward(self, b, ft, x):
        s_x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        h_x = self.sublayer[1](s_x, lambda s_x: self.attn[1](s_x, ft['encoded_his'], ft['encoded_his'], b.his_mask))
        
        if self.args.query_mm == 'query':
            c_x = self.sublayer[2](h_x, lambda h_x: self.attn[2](h_x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
            q_x = self.sublayer[3](c_x, lambda c_x: self.attn[3](c_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
            mm_x = q_x
        elif self.args.query_mm == 'caption':
            q_x = self.sublayer[2](h_x, lambda h_x: self.attn[2](h_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
            c_x = self.sublayer[3](q_x, lambda q_x: self.attn[3](q_x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
            mm_x = c_x

        self.attn_idx, self.ff_idx, self.sublayer_idx = 4, 0, 4        
        ts_x = self.temporal2spatial(b, ft, mm_x)        
        st_x = self.spatial2temporal(b, ft, mm_x)

        ts = ts_x + st_x        
        ts = self.sublayer[self.sublayer_idx](ts, self.ff[self.ff_idx])

        return ts
    
class MultimodalDecoderLayer8(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, nb_ff, dropout, args):
        super(MultimodalDecoderLayer8, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = clones(ff, nb_ff) 
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn+nb_ff)
        self.args = args 

    def forward(self, b, ft, x):
        s_x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        h_x = self.sublayer[1](s_x, lambda s_x: self.attn[1](s_x, ft['encoded_his'], ft['encoded_his'], b.his_mask))
        
        if self.args.query_mm == 'query':
            c_x = self.sublayer[2](h_x, lambda h_x: self.attn[2](h_x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
            q_x = self.sublayer[3](c_x, lambda c_x: self.attn[3](c_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
            mm_x = q_x
        elif self.args.query_mm == 'caption':
            q_x = self.sublayer[2](h_x, lambda h_x: self.attn[2](h_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
            c_x = self.sublayer[3](q_x, lambda q_x: self.attn[3](q_x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
            mm_x = c_x
        
        temporal = ft['temporal_ft']
        spatial = ft['spatial_ft']
                
        t_x = self.sublayer[4](mm_x, lambda mm_x: self.attn[4](mm_x, temporal, temporal, b.temporal_mask)) 
        t_x = self.sublayer[5](t_x, self.ff[0])

        s_x = self.sublayer[6](mm_x, lambda mm_x: self.attn[5](mm_x, spatial, spatial, b.spatial_mask))
        s_x = self.sublayer[7](s_x, self.ff[1])

        ts = t_x + s_x
        ts = self.sublayer[8](ts, self.ff[2])
                
        return ts
    
class MultimodalDecoder6(nn.Module):
    def __init__(self, layer, N):
        super(MultimodalDecoder6, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, b, ft, x):
        in_ft = {}
        in_ft['temporal'] = x
        in_ft['spatial'] = x
        for layer in self.layers:
            in_ft = layer(b, ft, in_ft)
        out = in_ft['temporal'] + in_ft['spatial']
        ft['decoded_text'] = self.norm(out)
        return ft
    
class MultimodalDecoderLayer9(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, nb_ff, dropout, args):
        super(MultimodalDecoderLayer9, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = clones(ff, nb_ff) 
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn+nb_ff)
        self.args = args 

    def forward(self, b, ft, in_ft):
        in_ft['temporal'] = self.temporal_decode(b, ft, in_ft['temporal'])
        in_ft['spatial'] = self.spatial_decode(b, ft, in_ft['spatial'])
        return in_ft
        
    def temporal_decode(self, b, ft, x):
        s_x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        h_x = self.sublayer[1](s_x, lambda s_x: self.attn[1](s_x, ft['encoded_his'], ft['encoded_his'], b.his_mask))
        
        if self.args.query_mm == 'query':
            c_x = self.sublayer[2](h_x, lambda h_x: self.attn[2](h_x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
            q_x = self.sublayer[3](c_x, lambda c_x: self.attn[3](c_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
            mm_x = q_x
        elif self.args.query_mm == 'caption':
            q_x = self.sublayer[2](h_x, lambda h_x: self.attn[2](h_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
            c_x = self.sublayer[3](q_x, lambda q_x: self.attn[3](q_x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
            mm_x = c_x
        
        temporal = ft['temporal_ft']
                
        t_x = self.sublayer[4](mm_x, lambda mm_x: self.attn[4](mm_x, temporal, temporal, b.temporal_mask)) 
        t_x = self.sublayer[5](t_x, self.ff[0])
        
        return t_x
    
    def spatial_decode(self, b, ft, x):
        s_x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        h_x = self.sublayer[1](s_x, lambda s_x: self.attn[1](s_x, ft['encoded_his'], ft['encoded_his'], b.his_mask))
        
        if self.args.query_mm == 'query':
            c_x = self.sublayer[2](h_x, lambda h_x: self.attn[2](h_x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
            q_x = self.sublayer[3](c_x, lambda c_x: self.attn[3](c_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
            mm_x = q_x
        elif self.args.query_mm == 'caption':
            q_x = self.sublayer[2](h_x, lambda h_x: self.attn[2](h_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
            c_x = self.sublayer[3](q_x, lambda q_x: self.attn[3](q_x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
            mm_x = c_x
        
        spatial = ft['spatial_ft']

        s_x = self.sublayer[6](mm_x, lambda mm_x: self.attn[5](mm_x, spatial, spatial, b.spatial_mask))
        s_x = self.sublayer[7](s_x, self.ff[1])
        
        return s_x

class MultimodalDecoder7(nn.Module):
    def __init__(self, layer, N):
        super(MultimodalDecoder7, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, b, ft, x):
        in_ft = {}
        in_ft['cap_temporal'] = x
        in_ft['cap_spatial'] = x
        in_ft['que_temporal'] = x
        in_ft['que_spatial'] = x
        for layer in self.layers:
            in_ft = layer(b, ft, in_ft)
        pdb.set_trace()
        out = in_ft['cap_temporal'] + in_ft['cap_spatial'] + in_ft['que_temporal'] + in_ft['que_spatial']
        ft['decoded_text'] = self.norm(out)
        return ft
    
class MultimodalDecoderLayer10(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, nb_ff, dropout, args):
        super(MultimodalDecoderLayer10, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = clones(ff, nb_ff) 
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn+nb_ff)
        self.args = args 

    def forward(self, b, ft, in_ft):
        in_ft['cap_temporal'] = self.cap_temporal_decode(b, ft, in_ft['cap_temporal'])
        in_ft['cap_spatial'] = self.cap_spatial_decode(b, ft, in_ft['cap_spatial'])
        in_ft['que_temporal'] = self.cap_temporal_decode(b, ft, in_ft['que_temporal'])
        in_ft['que_spatial'] = self.cap_spatial_decode(b, ft, in_ft['que_spatial'])
        return in_ft
        
    def cap_temporal_decode(self, b, ft, x):
        s_x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        h_x = self.sublayer[1](s_x, lambda s_x: self.attn[1](s_x, ft['encoded_his'], ft['encoded_his'], b.his_mask))

        c_x = self.sublayer[2](h_x, lambda h_x: self.attn[2](h_x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
        mm_x = c_x
        temporal = ft['temporal_ft']
        t_x = self.sublayer[3](mm_x, lambda mm_x: self.attn[3](mm_x, temporal, temporal, b.temporal_mask)) 
        t_x = self.sublayer[4](t_x, self.ff[0])        
        return t_x
    
    def que_temporal_decode(self, b, ft, x):
        s_x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        h_x = self.sublayer[1](s_x, lambda s_x: self.attn[1](s_x, ft['encoded_his'], ft['encoded_his'], b.his_mask))
        
        q_x = self.sublayer[5](h_x, lambda h_x: self.attn[4](h_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
        mm_x = q_x
        temporal = ft['temporal_ft']
        t_x = self.sublayer[6](mm_x, lambda mm_x: self.attn[5](mm_x, temporal, temporal, b.temporal_mask)) 
        t_x = self.sublayer[7](t_x, self.ff[1])        
        return t_x
    
    def cap_spatial_decode(self, b, ft, x):
        s_x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        h_x = self.sublayer[1](s_x, lambda s_x: self.attn[1](s_x, ft['encoded_his'], ft['encoded_his'], b.his_mask))
        
        c_x = self.sublayer[8](h_x, lambda h_x: self.attn[6](h_x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
        mm_x = c_x
        spatial = ft['spatial_ft']
        s_x = self.sublayer[9](mm_x, lambda mm_x: self.attn[7](mm_x, spatial, spatial, b.spatial_mask))
        s_x = self.sublayer[10](s_x, self.ff[2])
        return s_x
    
    def que_spatial_decode(self, b, ft, x):
        s_x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        h_x = self.sublayer[1](s_x, lambda s_x: self.attn[1](s_x, ft['encoded_his'], ft['encoded_his'], b.his_mask))

        q_x = self.sublayer[11](h_x, lambda h_x: self.attn[8](h_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
        mm_x = q_x
        spatial = ft['spatial_ft']
        s_x = self.sublayer[12](mm_x, lambda mm_x: self.attn[9](mm_x, spatial, spatial, b.spatial_mask))
        s_x = self.sublayer[13](s_x, self.ff[3])
        return s_x

class MultimodalDecoderLayer11(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, dropout, args):
        super(MultimodalDecoderLayer11, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = ff
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn+1)
        self.args = args 
                
    def forward(self, b, ft, x):
        s_x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        h_x = self.sublayer[1](s_x, lambda s_x: self.attn[1](s_x, ft['encoded_his'], ft['encoded_his'], b.his_mask)) 
        c_x = self.sublayer[2](h_x, lambda h_x: self.attn[2](h_x, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
        q_x = self.sublayer[3](c_x, lambda c_x: self.attn[3](c_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
        
        temporal = ft['temporal_ft']
        spatial = ft['spatial_ft']
        
        if self.args.vid_enc_mode in [15, 16]:
            if self.args.query_mm == 'query':
                temporal_mask = b.query_mask
                spatial_mask = b.query_mask
            elif self.args.query_mm == 'caption':
                temporal_mask = b.cap_mask
                spatial_mask = b.cap_mask
        else:
            temporal_mask = b.temporal_mask
            spatial_mask = b.spatial_mask
        
        t_x = self.sublayer[4](q_x, lambda q_x: self.attn[4](q_x, temporal, temporal, temporal_mask)) 
        s_x = self.sublayer[5](t_x, lambda t_x: self.attn[5](t_x, spatial, spatial, spatial_mask))
        out = self.sublayer[6](s_x, self.ff)

        return out
    
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
                #else:
                #    self.vc_combine_W = nn.Linear(v_layer.size*4, 3)

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
                #if layer == self.layers[-1]: 
                #    print('combine score s2t: {}'.format(combine_score.sum(1)[0][0]))
                #    print('combine score t2s: {}'.format(combine_score.sum(1)[0][1]))
                if self.args.t2s and self.args.s2t:
                    ft['encoded_ft'] = combine_score[:,:,0].unsqueeze(-1)*ft['temporal_ft'] + \
                        combine_score[:,:,1].unsqueeze(-1)*ft['spatial_ft']               
                if self.a_N>0:
                    ft['encoded_ft'] += combine_score[:,:,2].unsqueeze(-1)*ft['audio_ft']
            x = layer(b, ft, x)
            count+=1
            
        ft['decoded_text'] = self.norm(x)
        return ft
