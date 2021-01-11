import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable
from model.modules import * 
from model.encoder import *
from model.decoder import *
from model.generator import *
import pdb

class MTN(nn.Module):
    def __init__(self, args, text_encoder, vid_encoder, text_decoder, mutlimodal_decoder, 
                 query_embed, his_embed, cap_embed, tgt_embed, 
                 generator, ae_generator,
                ptr_gen=False):
        super(MTN, self).__init__() 
        self.text_encoder = text_encoder
        self.vid_encoder = vid_encoder
        self.text_decoder = text_decoder
        self.mutlimodal_decoder = mutlimodal_decoder
        self.query_embed = query_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.ae_generator = ae_generator
        self.ptr_gen = ptr_gen
        self.args = args 

    def forward(self, b):
        ft = self.encode(b)
        ft = self.decode(b, ft)
        return ft 
    
    def encode(self, b):
        ft = {}
        ft = self.encode_text(b, ft)
        ft = self.encode_vid(b, ft)
        return ft 
    
    def encode_text(self, b, ft):
        encoded_query, encoded_cap, encoded_his = self.text_encoder(self.query_embed(b.query), self.query_embed(b.cap), self.query_embed(b.his))
        ft['encoded_query'] = encoded_query
        ft['encoded_cap'] = encoded_cap
        ft['encoded_his'] = encoded_his
        return ft 
        
    def encode_vid(self, b, ft):
        ft = self.vid_encoder(b, ft) 
        return ft 
    
    def decode(self, b, ft):
        ft = self.multimodal_decode_text(b, ft)
        return ft
    
    def multimodal_decode_text(self, b, ft):
        encoded_tgt = self.tgt_embed(b.trg)
        ft['encoded_tgt'] = encoded_tgt
        ft = self.mutlimodal_decoder(b, ft, encoded_tgt)
        return ft 
    
def make_model(src_vocab, tgt_vocab, args, ft_sizes=None, embeddings=None):  
    
    N=args.nb_blocks
    venc_N=args.nb_venc_blocks
    cenc_N=args.nb_cenc_blocks
    aenc_N=args.nb_aenc_blocks
    d_model=args.d_model
    d_ff = d_model * 4
    h=args.att_h
    dropout=args.dropout
    ptr_gen=args.ptr_gen
    ptr_ft=args.ptr_ft
    
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    query_embed = [Embeddings(d_model, src_vocab), c(position)]
    query_embed = nn.Sequential(*query_embed)
    tgt_embed = query_embed
        
    if ptr_gen:
        if len(ptr_ft.split(','))>1:
            pointer_attn = nn.ModuleList()
            ptr_ft_ls = ptr_ft.split(',')
            for ft in ptr_ft_ls:
                pointer_attn.append(MultiHeadedAttention(1, d_model, dropout=0))
            generator = MultiPointerGenerator(d_model, tgt_embed[0].lut.weight, pointer_attn, len(ptr_ft_ls))
        else:
            pointer_attn = MultiHeadedAttention(1, d_model, dropout=0)
            generator = PointerGenerator(d_model, tgt_embed[0].lut.weight, pointer_attn)
    else:
        if separate_out_linear:
            generator=Generator(d_model, tgt_vocab)
        else:
            generator=Generator(d_model, tgt_vocab, tgt_embed[0].lut.weight)
            
    if args.auto_encoder:
        ae_generator = Generator(d_model, tgt_vocab, tgt_embed[0].lut.weight) 
    else:
        ae_generator = None
    
    his_embed = None 
    cap_embed = None 
    text_encoder=Encoder(d_model, nb_layers=3)
    vid_position = None
    
    if len(ft_sizes)>0:
        vid_W =  nn.Linear(ft_sizes[0], d_model)
        if len(ft_sizes)>1:
            a_W = nn.Linear(ft_sizes[1], d_model)
        else:
            a_W = None 
    else:
        vid_W = None 
    
    if args.vid_enc_mode == 22:
        vid_encoder = VidEncoder8(c(vid_W), c(a_W), vid_position,  venc_N, aenc_N, d_model, args)
        text_decoder = None
        
        if args.t2s==0 or args.s2t==0:
            nb_v_attn=3
            nb_ff=1
        else:
            nb_v_attn=6
            nb_ff=2
        v_layer = VidEncoderLayer4(d_model, c(attn), nb_v_attn, c(ff), nb_ff, dropout, args)
        c_layer = CapEncoderLayer(d_model, c(attn), 2, c(ff), dropout)
        a_layer = AudioEncoderLayer(d_model, c(attn), 2, c(ff), dropout)
        
        nb_attn=3
        if args.nb_cenc_blocks>0 and args.nb_venc_blocks>0 and args.enc_vc_combine != 'none':
            nb_attn+=1 
        else:
            if args.nb_cenc_blocks>0:
                nb_attn+=1 
            if args.nb_aenc_blocks>0:
                nb_attn+=1
            if args.nb_venc_blocks>0:
                if args.enc_st_combine in ['dyn', 'sum', 'early_sum', 'early_dyn'] and args.s2t and args.t2s:
                    nb_attn+=1
                else:
                    nb_attn+=2
        mm_layer = MultimodalDecoderLayer12(d_model, c(attn), nb_attn, c(ff), dropout, args)
        mutlimodal_decoder = MultimodalDecoder8(v_layer, c_layer, a_layer, mm_layer, venc_N, cenc_N, aenc_N, N, args)
        
    model = MTN(
          args=args,
          text_encoder=text_encoder, 
          vid_encoder=vid_encoder,
          text_decoder=text_decoder,
          mutlimodal_decoder=mutlimodal_decoder,
          query_embed=query_embed,
          his_embed=his_embed,
          cap_embed=cap_embed,
          tgt_embed=tgt_embed,
          generator=generator,
          ae_generator=ae_generator,
          ptr_gen = ptr_gen)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model
