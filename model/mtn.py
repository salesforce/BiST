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
                 query_embed, tgt_embed, 
                 generator, ae_generator,
                 prior=None):
        super(MTN, self).__init__() 
        self.text_encoder = text_encoder
        self.vid_encoder = vid_encoder
        self.mutlimodal_decoder = mutlimodal_decoder
        self.query_embed = query_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.ae_generator = ae_generator
        self.prior = prior 
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
        encoded_query = self.text_encoder(self.query_embed(b.query))[0]
        ft['encoded_query'] = encoded_query
        return ft 
        
    def encode_vid(self, b, ft):
        ft = self.vid_encoder(b, ft) 
        return ft 
    
    def decode(self, b, ft):
        ft = self.multimodal_decode_text(b, ft)
        return ft
    
    def multimodal_decode_text(self, b, ft):
        encoded_tgt = self.prior.unsqueeze(0).expand(b.query.shape[0], self.prior.shape[0]).unsqueeze(1) 
        ft['encoded_tgt'] = encoded_tgt
        ft = self.mutlimodal_decoder(b, ft, encoded_tgt)
        return ft 

def make_model(src_vocab, tgt_vocab, args, ft_sizes=None, embeddings=None):  
    N=args.nb_blocks
    venc_N=args.nb_venc_blocks
    d_model=args.d_model
    d_ff = d_model * 4
    h=args.att_h
    dropout=args.dropout
    
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    query_embed = [Embeddings(d_model, src_vocab), c(position)]
    query_embed = nn.Sequential(*query_embed)
    tgt_embed = query_embed
    
    if args.task == 'FrameQA':
        generator = Generator(d_model, tgt_vocab)
    else:
        generator = RegressGenerator(d_model)
    
    if args.auto_encoder:
        ae_generator = Generator(d_model, None, tgt_embed[0].lut.weight) 
    else:
        ae_generator = None
    
    text_encoder=Encoder(d_model, nb_layers=1)
    vid_position = None
    if len(ft_sizes)>0:
        vid_W =  nn.Linear(ft_sizes[0], d_model)
        if len(ft_sizes)>1:
            a_W = nn.Linear(ft_sizes[1], d_model)
        else:
            a_W = None 
    else:
        vid_W = None 
    
    if True:
        vid_encoder = VidEncoder8(c(vid_W), c(a_W), vid_position,  venc_N, 0, d_model, args)
        text_decoder = None
        
        if args.t2s==0 or args.s2t==0:
            nb_v_attn=3
            nb_ff=1
        else:
            nb_v_attn=6
            nb_ff=2
        v_layer = VidEncoderLayer4(d_model, c(attn), nb_v_attn, c(ff), nb_ff, dropout, args)
        
        nb_attn=2
        if args.enc_vc_combine != 'none':
            nb_attn+=1 
        else:
            if args.nb_venc_blocks>0:
                if args.enc_st_combine in ['dyn', 'sum', 'early_sum', 'early_dyn'] and args.s2t and args.t2s:
                    nb_attn+=1
                else:
                    nb_attn+=2
                    
        if args.prior:
            prior = Variable(torch.zeros(d_model), requires_grad=True).cuda()
        else:
            prior = None 
            
        mm_layer = MultimodalDecoderLayer12(d_model, c(attn), nb_attn, c(ff), dropout, args)
        mutlimodal_decoder = MultimodalDecoder8(v_layer, None, None, mm_layer, venc_N, 0, 0, N, args)
        
    model = MTN(
          args=args,
          text_encoder=text_encoder, 
          vid_encoder=vid_encoder,
          text_decoder=text_decoder,
          mutlimodal_decoder=mutlimodal_decoder,
          query_embed=query_embed,
          tgt_embed=tgt_embed,
          generator=generator,
          ae_generator=ae_generator,
          prior=prior)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    
    return model
