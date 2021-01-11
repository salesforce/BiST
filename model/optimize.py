import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy, time
import pdb 

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * \
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, ae_generator, criterion, opt=None, l=1.0, args=None):
        self.generator = generator
        self.ae_generator= ae_generator
        self.criterion = criterion
        self.opt = opt 
        self.l = l
        self.args = args
    
    def __call__(self, ft, batch):
        loss = 0
        norm = batch.ntokens
        out = self.generator(ft, batch, self.args)
        out_loss = self.criterion(out.contiguous().view(-1, out.size(-1)), batch.trg_y.contiguous().view(-1)) / norm.float()
        loss += out_loss
        
        temporal_ae_loss, spatial_ae_loss = 0.0, 0.0 
        ae_norm = batch.qntokens
        if self.args.auto_encoder:
            if self.args.nb_cenc_blocks>0:
                cap_ae_out = self.ae_generator(ft, batch, self.args, 'cap_ft')
                cap_ae_loss = self.criterion(cap_ae_out.contiguous().view(
                    -1, out.size(-1)), batch.query.contiguous().view(-1)) / ae_norm.float()
                loss += cap_ae_loss
            if self.args.nb_aenc_blocks>0:
                audio_ae_out = self.ae_generator(ft, batch, self.args, 'audio_ft')
                audio_ae_loss = self.criterion(audio_ae_out.contiguous().view(
                    -1, out.size(-1)), batch.query.contiguous().view(-1)) / ae_norm.float()
                loss += audio_ae_loss
            if self.args.nb_venc_blocks>0:
                if self.args.enc_st_combine == 'none':
                    if self.args.s2t:
                        temporal_ae_out = self.ae_generator(ft, batch, self.args, 'temporal_ft')
                        temporal_ae_loss = self.criterion(temporal_ae_out.contiguous().view(
                            -1, out.size(-1)), batch.query.contiguous().view(-1)) / ae_norm.float()
                        loss += temporal_ae_loss
                    if self.args.t2s:
                        spatial_ae_out = self.ae_generator(ft, batch, self.args, 'spatial_ft') 
                        spatial_ae_loss = self.criterion(spatial_ae_out.contiguous().view(
                            -1, out.size(-1)), batch.query.contiguous().view(-1)) / ae_norm.float()
                        loss += spatial_ae_loss
                else:
                    spatiotemporal_ae_out = self.ae_generator(ft, batch, self.args, 'spatiotemporal_ft')
                    spatiotemporal_ae_loss = self.criterion(spatiotemporal_ae_out.contiguous().view(
                        -1, out.size(-1)), batch.query.contiguous().view(-1)) / ae_norm.float()
                    loss += spatiotemporal_ae_loss
            
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
            
        losses = {}
        losses['out'] = out_loss.item() * norm.float()
        losses['temporal_ae'] = temporal_ae_loss * ae_norm.float()
        losses['spatial_ae'] = spatial_ae_loss * ae_norm.float()
        
        return losses
    
