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
        if hasattr(self, 'args') and self.args.vid_enc_mode in [5, 7, 8, 9, 10, 11, 12, 14, 16, 19, 20, 21, 22]:
            ft = self.vid_encoder(b, ft) 
        else:
            if self.args.vid_enc_mode in [2, 3, 4, 13, 15, 17, 18]:
                if self.args.query_mm == 'query':
                    query = ft['encoded_query']
                    query_mask = b.query_mask
                elif self.args.query_mm == 'caption':
                    query = ft['encoded_cap']
                    query_mask = b.cap_mask 
            else:
                query = ft['encoded_query'].mean(1).unsqueeze(1)
            ft = self.vid_encoder(b, query, query_mask, ft)
        return ft 
    
    def decode(self, b, ft):
        if hasattr(self, 'args') and \
            self.args.vid_enc_mode in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
            ft = self.multimodal_decode_text(b, ft)
        else:
            ft = self.decode_text(b, ft) 
            ft = self.multimodal_decode_text(b, ft)
        return ft
    
    def decode_text(self, b, ft):
        encoded_tgt = self.tgt_embed(b.trg)
        ft['encoded_tgt'] = encoded_tgt
        ft = self.text_decoder(b, ft, encoded_tgt) 
        return ft 
    
    def multimodal_decode_text(self, b, ft):
        if hasattr(self, 'args') and \
            self.args.vid_enc_mode in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
            encoded_tgt = self.tgt_embed(b.trg)
            ft['encoded_tgt'] = encoded_tgt
            ft = self.mutlimodal_decoder(b, ft, encoded_tgt)
        else:
            ft = self.mutlimodal_decoder(b, ft) 
        return ft 
    

def make_model(src_vocab, tgt_vocab, args, ft_sizes=None, embeddings=None):  
    
    N=args.nb_blocks
    venc_N=args.nb_venc_blocks
    cenc_N=args.nb_cenc_blocks
    aenc_N=args.nb_aenc_blocks
    d_model=args.d_model
    #d_ff=args.d_ff
    d_ff = d_model * 4
    h=args.att_h
    dropout=args.dropout
    separate_his_embed=args.separate_his_embed
    separate_cap_embed=args.separate_cap_embed 
    separate_out_embed=args.separate_out_embed
    separate_out_linear=args.separate_out_linear
    diff_encoder=args.diff_encoder
    diff_embed=args.diff_embed
    diff_gen=args.diff_gen
    auto_encoder_ft=args.auto_encoder_ft
    ptr_gen=args.ptr_gen
    ptr_ft=args.ptr_ft
    
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    query_embed = [Embeddings(d_model, src_vocab), c(position)]
    query_embed = nn.Sequential(*query_embed)
    
    if not separate_out_embed:
        tgt_embed = query_embed
    else:
        tgt_embed = [Embeddings(d_model, tgt_vocab), c(position)]
        tgt_embed = nn.Sequential(*tgt_embed)  
        
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
    
    if separate_his_embed:
        his_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    else:
        his_embed = None 
        
    if separate_cap_embed:
        cap_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    else:
        cap_embed = None 
        
    text_encoder=Encoder(d_model, nb_layers=3)
    if args.vid_pos:
        vid_position = c(position)
    else:
        vid_position = None
    if len(ft_sizes)>0:
        vid_W =  nn.Linear(ft_sizes[0], d_model)
        if len(ft_sizes)>1:
            a_W = nn.Linear(ft_sizes[1], d_model)
        else:
            a_W = None 
    else:
        vid_W = None 
    
    if args.vid_enc_mode == 2:
        vid_encoder = VidEncoder2(vid_W, vid_position, VidEncoderLayer2(d_model, c(attn), 2, c(ff), c(ff), dropout), N)
        text_decoder = Decoder(DecoderLayer(d_model, c(attn), 4, c(ff), dropout), N)
        mutlimodal_decoder = MultimodalDecoder2(MultimodalDecoderLayer2(d_model, c(attn), 1, c(ff), c(ff), dropout), N, args.dec_st_combine)
    elif args.vid_enc_mode == 3: 
        vid_encoder = VidEncoder2(vid_W, vid_position, VidEncoderLayer2(d_model, c(attn), 2, c(ff), c(ff), dropout), N)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder3(MultimodalDecoderLayer3(d_model, c(attn), 4, c(attn), 1, c(ff), c(ff), dropout, args), N)
    elif args.vid_enc_mode == 4:
        vid_encoder = VidEncoder2(vid_W, vid_position, VidEncoderLayer3(d_model, c(attn), 2, c(ff), c(ff), dropout), N)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder3(MultimodalDecoderLayer3(d_model, c(attn), 4, c(attn), 1, c(ff), c(ff), dropout, args), N)
    elif args.vid_enc_mode == 5:
        vid_encoder = VidEncoder3(d_model, c(vid_W), c(vid_W), vid_position)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder4(MultimodalDecoderLayer4(
            d_model, c(attn), 4, c(attn), 1, c(attn), 2, 
            c(ff), c(ff), c(ff), c(ff), dropout, args), N)
    elif args.vid_enc_mode == 6:
        vid_encoder = VidEncoder(c(vid_W), c(vid_W), vid_position, VidEncoderLayer(d_model, c(attn), 3, c(ff), c(ff), dropout, args), N)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder3(MultimodalDecoderLayer3(d_model, c(attn), 4, c(attn), 1, c(ff), c(ff), dropout, args), N)
    elif args.vid_enc_mode == 7:
        vid_encoder = VidEncoder3(d_model, c(vid_W), c(vid_W), vid_position)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder5(MultimodalDecoderLayer5(d_model, c(attn), 4, c(attn), 2, c(ff), c(ff), c(ff), c(ff), c(ff), dropout, args), N)
    elif args.vid_enc_mode == 8:
        vid_encoder = VidEncoder4(d_model, vid_W, vid_position)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder5(MultimodalDecoderLayer6(d_model, c(attn), 12, c(ff), 7, 19, dropout, args), N)
    elif args.vid_enc_mode == 9:
        vid_encoder = VidEncoder4(d_model, vid_W, vid_position)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder5(MultimodalDecoderLayer7(d_model, c(attn), 8, c(ff), 3, 11, dropout, args), N)
    elif args.vid_enc_mode == 10:
        vid_encoder = VidEncoder3(d_model, c(vid_W), c(vid_W), vid_position)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder5(MultimodalDecoderLayer8(d_model, c(attn), 6, c(ff),3 ,dropout, args), N)
    elif args.vid_enc_mode == 11:
        vid_encoder = VidEncoder3(d_model, c(vid_W), c(vid_W), vid_position)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder6(MultimodalDecoderLayer9(d_model, c(attn), 6, c(ff),3 ,dropout, args), N)
    elif args.vid_enc_mode == 12:
        vid_encoder = VidEncoder3(d_model, c(vid_W), c(vid_W), vid_position)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder7(MultimodalDecoderLayer10(d_model, c(attn), 6, c(ff),2 ,dropout, args), N)
    elif args.vid_enc_mode == 13:
        vid_encoder = VidEncoder2(c(vid_W), c(vid_W), vid_position, VidEncoderLayer3(d_model, c(attn), 2, c(ff), c(ff), dropout), N)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder5(MultimodalDecoderLayer5(d_model, c(attn), 4, c(attn), 2, c(ff), c(ff), c(ff), c(ff), c(ff), dropout, args), N)
    elif args.vid_enc_mode == 14:
        vid_encoder = VidEncoder5(c(vid_W), vid_position, VidEncoderLayer4(d_model, c(attn), 6, c(ff), 2, dropout), venc_N)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder5(MultimodalDecoderLayer5(d_model, c(attn), 4, c(attn), 2, c(ff), c(ff), c(ff), c(ff), c(ff), dropout, args), N)
    elif args.vid_enc_mode == 15:
        vid_encoder = VidEncoder2(c(vid_W), c(vid_W), vid_position, VidEncoderLayer3(d_model, c(attn), 2, c(ff), c(ff), dropout), N)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder5(MultimodalDecoderLayer11(d_model, c(attn), 6, c(ff), dropout, args), N)
    elif args.vid_enc_mode == 16:
        vid_encoder = VidEncoder5(c(vid_W), vid_position, VidEncoderLayer4(d_model, c(attn), 6, c(ff), 2, dropout), venc_N)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder5(MultimodalDecoderLayer11(d_model, c(attn), 6, c(ff), dropout, args), N)
    elif args.vid_enc_mode == 17:
        vid_encoder = VidEncoder2a(c(vid_W), c(vid_W), vid_position, VidEncoderLayer3(d_model, c(attn), 2, c(ff), c(ff), dropout), N)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder5(MultimodalDecoderLayer5(d_model, c(attn), 4, c(attn), 2, c(ff), c(ff), c(ff), c(ff), c(ff), dropout, args), N)
    elif args.vid_enc_mode == 18: # TODO
        ts_ff = PositionwiseFeedForward(d_model, d_ff, dropout, int(d_model/2))
        vid_encoder = VidEncoder2a(c(vid_W), c(vid_W), vid_position, VidEncoderLayer3(d_model, c(attn), 2, c(ff), c(ff), dropout), N)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder5(MultimodalDecoderLayer5a(d_model, c(attn), 4, c(attn), 2, c(ts_ff), c(ts_ff), c(ts_ff), c(ts_ff), c(ff), dropout, args), N)
    elif args.vid_enc_mode == 19:
        vid_encoder = VidEncoder5a(c(vid_W), vid_position, VidEncoderLayer4(d_model, c(attn), 6, c(ff), 2, dropout), venc_N)
        text_decoder = None
        mutlimodal_decoder = MultimodalDecoder5(MultimodalDecoderLayer5(d_model, c(attn), 4, c(attn), 2, c(ff), c(ff), c(ff), c(ff), c(ff), dropout, args), N)
    elif args.vid_enc_mode == 20:
        if args.t2s==0 or args.s2t==0:
            nb_v_attn=3
            nb_ff=1
        else:
            nb_v_attn=6
            nb_ff=2
        v_layer = VidEncoderLayer4(d_model, c(attn), nb_v_attn, c(ff), nb_ff, dropout, args)
        c_layer = CapEncoderLayer(d_model, c(attn), 2, c(ff), dropout)
        a_layer = AudioEncoderLayer(d_model, c(attn), 2, c(ff), dropout)
        vid_encoder = VidEncoder6(c(vid_W), vid_position, v_layer, c_layer, venc_N, cenc_N, args, a_W, a_layer, aenc_N)
        text_decoder = None
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
        mutlimodal_decoder = MultimodalDecoder5(MultimodalDecoderLayer12(d_model, c(attn), nb_attn, c(ff), dropout, args), N)
    elif args.vid_enc_mode == 21:
        v_layer = VidEncoderLayer5(d_model, c(attn), 2, c(ff), dropout)
        c_layer = CapEncoderLayer(d_model, c(attn), 2, c(ff), dropout)
        vid_encoder = VidEncoder7(c(vid_W), vid_position, v_layer, c_layer, venc_N, cenc_N, args)
        text_decoder = None
        nb_attn=4
        if args.nb_venc_blocks>0:
            nb_attn+=1
        mutlimodal_decoder = MultimodalDecoder5(MultimodalDecoderLayer12(d_model, c(attn), nb_attn, c(ff), dropout, args), N)
    elif args.vid_enc_mode == 22:
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
    else:
        vid_encoder = VidEncoder(vid_W, vid_position, VidEncoderLayer(d_model, c(attn), 3, c(ff), c(ff), dropout, args), N)
        text_decoder = Decoder(DecoderLayer(d_model, c(attn), 4, c(ff), dropout), N)
        mutlimodal_decoder = MultimodalDecoder(MultimodalDecoderLayer(d_model, c(attn), 2, c(ff), c(ff), dropout), N, args.dec_st_combine)
        
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

    if embeddings is not None: 
        if args.fixed_word_emb:
            query_embed[0].lut = query_embed[0].lut.from_pretrained(torch.tensor(embeddings).float(), freeze=True)
        else:
            query_embed[0].lut = query_embed[0].lut.from_pretrained(torch.tensor(embeddings).float(), freeze=False)
        if embeddings.shape[1] != d_model:
            new_embed = [query_embed[0], nn.Linear(embeddings.shape[1], d_model), c(position)]
        else:
            new_embed = [query_embed[0], c(position)]
        query_embed = nn.Sequential(*new_embed)
        tgt_embed = query_embed
        model.query_embed = query_embed
        model.tgt_embed = tgt_embed
    return model
