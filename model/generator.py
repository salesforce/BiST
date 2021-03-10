import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable
from model.modules import *
import pdb 


class RegressGenerator(nn.Module):
    def __init__(self, d_model):
        super(RegressGenerator, self).__init__()
        self.proj = nn.Linear(d_model, 1)
        
    def forward(self, x):
        return self.proj(x)
    
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
    def forward(self, ft, batch, args, ft_key='decoded_text', log_softmax=True):
        x = ft[ft_key]
        if self.shared_W:
            out = x.matmul(self.proj.transpose(1,0))
            if log_softmax:
                return F.log_softmax(out, dim=-1)
            else:
                return out
        else:
            if log_softmax:
                return F.log_softmax(self.proj(x), dim=-1)
            else:
                return self.proj(x) 
