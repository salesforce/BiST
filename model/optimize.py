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
    def __init__(self, generator, ae_generator, criterion, ae_criterion, opt=None, args=None):
        self.generator = generator
        self.ae_generator= ae_generator
        self.criterion = criterion
        self.ae_criterion = ae_criterion 
        self.opt = opt 
        self.args = args
    
    def __call__(self, ft, batch, keep_average=False):
        loss = 0
        norm = batch.ntokens
        if self.args.task == 'FrameQA':
            out = self.generator(ft, batch, self.args, 'decoded_text', log_softmax=False)
            out_loss = self.criterion(out.squeeze(1), batch.trg)
        else:
            out = self.generator(ft['decoded_text'])
            if self.args.task == 'Count':
                out_loss = self.criterion(out.squeeze(), batch.trg.float())
            elif self.args.task in ['Trans', 'Action']:
                out_loss = self.criterion(out.squeeze().reshape(len(batch.trg), -1), batch.trg)
        loss += out_loss
        
        if self.args.task == 'Count':
            prediction = torch.clamp(torch.round(out.detach()), min=1, max=10).int()
            nb_correct = (prediction.squeeze() == batch.trg.int()).sum().item()
        elif self.args.task in ['Trans', 'Action']:
            _, prediction = torch.max(out.squeeze().reshape(len(batch.trg), -1), 1)
            nb_correct = (prediction == batch.trg.long()).sum().item()
        elif self.args.task == 'FrameQA':
            _, prediction = F.softmax(out, dim=-1).squeeze(1).max(dim=1)
            nb_correct = (prediction == batch.trg.long()).sum().item()
        
        temporal_ae_loss, spatial_ae_loss = 0.0, 0.0 
        ae_norm = batch.qntokens
        if self.args.auto_encoder:
            if self.args.nb_venc_blocks>0:
                if self.args.enc_st_combine == 'none':
                    if self.args.s2t:
                        temporal_ae_out = self.ae_generator(ft, batch, self.args, 'temporal_ft')
                        temporal_ae_loss = self.ae_criterion(temporal_ae_out.contiguous().view(
                            -1, temporal_ae_out.size(-1)), batch.query.contiguous().view(-1)) / ae_norm.float()
                        loss += temporal_ae_loss
                    if self.args.t2s:
                        spatial_ae_out = self.ae_generator(ft, batch, self.args, 'spatial_ft') 
                        spatial_ae_loss = self.ae_criterion(spatial_ae_out.contiguous().view(
                            -1, spatial_ae_out.size(-1)), batch.query.contiguous().view(-1)) / ae_norm.float()
                        loss += spatial_ae_loss
                else:
                    spatiotemporal_ae_out = self.ae_generator(ft, batch, self.args, 'spatiotemporal_ft')
                    spatiotemporal_ae_loss = self.ae_criterion(spatiotemporal_ae_out.contiguous().view(
                        -1, spatiotemporal_ae_out.size(-1)), batch.query.contiguous().view(-1)) / ae_norm.float()
                    loss += spatiotemporal_ae_loss
            
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
            
        losses = {}
        if keep_average:
            losses['out'] = out_loss.item() 
            losses['temporal_ae'] = temporal_ae_loss 
            losses['spatial_ae'] = spatial_ae_loss 
            acc = nb_correct/norm 
            return losses, acc, prediction.squeeze()
        else:
            losses['out'] = out_loss.item() * norm
            losses['temporal_ae'] = temporal_ae_loss * ae_norm.float()
            losses['spatial_ae'] = spatial_ae_loss * ae_norm.float()
            return losses, nb_correct, prediction.squeeze()
    
