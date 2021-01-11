#!/usr/bin/env python
import math
import sys
import time
import os
import json
import numpy as np
import pickle as pkl
import threading
import pdb 
from tqdm import tqdm 
import torch
import torch.nn as nn

from configs.train_configs import *
from model.mtn import *
from model.label_smoothing import * 
from model.optimize import *
import data.data_handler as dh

def run_epoch(data, loader, vocab, epoch, model, loss_compute, eval=False, gen_valid_indices=None):
    "Standard Training and Logging Function"
    total_tokens = 0 
    total_qtokens = 0
    total_loss = 0
    total_ae_temporal_loss = 0
    total_ae_spatial_loss = 0 
    it = tqdm(enumerate(loader),total=len(loader), desc="epoch {}/{}".format(epoch+1, args.num_epochs), ncols=0)
    for j, batch in it:  
        batch.move_to_cuda()
        out = model.forward(batch) 
        losses = loss_compute(out, batch)
        total_loss += losses['out']
        total_ae_temporal_loss += losses['temporal_ae']
        total_ae_spatial_loss += losses['spatial_ae']
        total_tokens += batch.ntokens
        total_qtokens += batch.qntokens
        if (j+1) % args.report_interval == 0 and not eval:
            print("Epoch: %d Step: %d Loss: %f AETemporalLoss: %f AESpatialLoss: %f" %
                    (epoch+1,j+1, losses['out']/batch.ntokens.float(), 
                     losses['temporal_ae']/batch.qntokens.float(), 
                     losses['spatial_ae']/batch.qntokens.float()))
            with open(train_log_path, "a") as f:
                f.write("{},{},{:e},{:e},{:e}\n".format(
                    epoch+1,j+1,losses['out']/batch.ntokens.float(), 
                    losses['temporal_ae']/batch.qntokens.float(), 
                    losses['spatial_ae']/batch.qntokens.float()))
    out_losses={}
    out_losses['out']=total_loss/total_tokens.float()
    out_losses['temporal_ae']=total_ae_temporal_loss/total_qtokens.float()
    out_losses['spatial_ae']=total_ae_spatial_loss//total_qtokens.float()
    return out_losses
                  
# get vocabulary
logging.info('Extracting words from ' + args.train_set)
if args.ptr_gen and args.cutoff<=0:
    train_vocab = dh.get_vocabulary(args.train_set, include_caption=args.include_caption, ptr_gen=args.ptr_gen)
    valid_vocab = dh.get_vocabulary(args.valid_set, include_caption=args.include_caption, ptr_gen=args.ptr_gen)
    test_vocab = dh.get_vocabulary(args.test_set, include_caption=args.include_caption, ptr_gen=args.ptr_gen)
    vocab = dh.merge_vocab([train_vocab, valid_vocab, test_vocab])
else:
    vocab = dh.get_vocabulary(args.train_set, include_caption=args.include_caption, cutoff=args.cutoff)
if args.word_emb != 'none':
    embeddings = dh.get_pretrained_emb(vocab, args.word_emb)
else:
    embeddings = None
# load data
logging.info('Loading training data from ' + args.train_set)
train_data = dh.load(args.fea_type, args.train_path, args.train_set, 
                     include_caption=args.include_caption, separate_caption=args.separate_caption,
                     vocab=vocab, max_history_length=args.max_history_length, 
                     merge_source=args.merge_source, skip=args.skip)
logging.info('Loading validation data from ' + args.valid_set)
valid_data = dh.load(args.fea_type, args.valid_path, args.valid_set, 
                     include_caption=args.include_caption, separate_caption=args.separate_caption, 
                     vocab=vocab, max_history_length=args.max_history_length, 
                     merge_source=args.merge_source, skip=args.skip)
if args.fea_type[0] == 'none':
    feature_dims = []
else:
    feature_dims = dh.feature_shape(train_data)
logging.info("Detected feature dims: {}".format(feature_dims));
logging.info('#vocab = %d' % len(vocab))
# make batchset for training
train_dataloader, train_samples = dh.create_dataset(train_data, args.batch_size, True, 
                                  include_caption=args.include_caption, separate_caption=args.separate_caption,
                                  cut_a=args.cut_a, num_workers=args.num_workers)
logging.info('#train sample = %d' % train_samples)
logging.info('#train batch = %d' % len(train_dataloader))
# make batchset for validation
valid_dataloader, valid_samples = dh.create_dataset(valid_data, args.batch_size, False, 
                                  include_caption=args.include_caption, separate_caption=args.separate_caption,
                                  cut_a=False, num_workers=args.num_workers)

gen_valid_indices = None
logging.info('#validation sample = %d' % valid_samples)
logging.info('#validation batch = %d' % len(valid_dataloader))
# create_model
model = make_model(len(vocab), len(vocab), args, ft_sizes=feature_dims, embeddings=embeddings)
model.cuda()
criterion = LabelSmoothing(size=len(vocab), padding_idx=vocab['<blank>'], smoothing=0.1)
criterion.cuda()	

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    device_ids = [int(i) for i in (args.device).split(',')]
    model = nn.DataParallel(model, device_ids=device_ids)
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
# save meta parameters
path = args.model + '.conf'
with open(path, 'wb') as f:
    pkl.dump((vocab, args), f, -1)
path2 = args.model + '_params.txt'
with open(path2, "w") as f: 
    for arg in vars(args):
        f.write("{}={}\n".format(arg, getattr(args, arg)))

logging.info('----------------')
logging.info('Start training')
logging.info('----------------')
# initialize status parameters
modelext = '.pth.tar'
min_valid_loss = 1.0e+10
bestmodel_num = 0
# save results 
trace_log_path = args.model+'_trace.csv'
with open(trace_log_path, "w") as f:
    f.write('epoch,split,loss,ae_temporal_loss,ae_spatial_loss\n') 
train_log_path = args.model+'_train.csv'
with open(train_log_path, "w") as f:  
    f.write('epoch,step,loss,ae_temporal_loss,ae_spatial_loss\n') 
print("Saving training results to {}".format(train_log_path))
print("Saving val results to {}".format(trace_log_path))   
model_opt = NoamOpt(args.d_model, 1, args.warmup_steps,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

#if torch.cuda.device_count() > 1:
#    train_loss = MultiGPULossCompute(model.generator, model.ae_generator, 
#                                     criterion, opt=model_opt, l=args.loss_l, args=args,
#                                    devices=devices)
#else:
train_loss = SimpleLossCompute(model.generator, model.ae_generator,
          criterion, opt=model_opt, l=args.loss_l, args=args)
val_loss = SimpleLossCompute(model.generator, model.ae_generator,
            criterion,opt=None, l=args.loss_l, args=args)

for epoch in range(args.num_epochs):
    # start training 
    model.train()
    train_losses = run_epoch(train_data, train_dataloader, vocab, epoch, model, train_loss)
    logging.info("epoch: {} train loss: {} aeTemporalLoss {} aeSpatialLoss {}".format(
        epoch+1, train_losses['out'], train_losses['temporal_ae'], train_losses['spatial_ae']))
    # test on validation data 
    logging.info('-------validation--------')
    model.eval()
    valid_losses = run_epoch(valid_data, valid_dataloader, vocab, epoch, model, val_loss,
                eval=True, gen_valid_indices=gen_valid_indices)
    logging.info("epoch: {} valid loss: {} aeTemporalLoss {} aeSpatialLoss {}".format(
        epoch+1, valid_losses['out'], valid_losses['temporal_ae'], valid_losses['spatial_ae']))              
    with open(trace_log_path,"a") as f:
        f.write("{},train,{:e},{:e},{:e}\n".format(
            epoch+1,train_losses['out'], train_losses['temporal_ae'], train_losses['spatial_ae']))
        f.write("{},val,{:e},{:e},{:e}\n".format(
            epoch+1,valid_losses['out'], valid_losses['temporal_ae'], valid_losses['spatial_ae']))        
    # update the model and save checkpoints
    valid_loss = valid_losses['out'] + valid_losses['temporal_ae'] + valid_losses['spatial_ae']
    if args.save_all:
        modelfile = args.model + '_' + str(epoch + 1) + modelext
        logging.info('writing model params to ' + modelfile)
        torch.save(model, modelfile)
    if min_valid_loss > valid_loss:
        bestmodel_num = epoch+1
        logging.info('validation loss reduced %.4f -> %.4f' % (min_valid_loss, valid_loss))
        min_valid_loss = valid_loss
        if args.save_all:
            best_modelfile = args.model + '_best' + modelext
            logging.info('a symbolic link is made as ' + best_modelfile)
            if os.path.exists(best_modelfile):
                os.remove(best_modelfile)
            os.symlink(os.path.basename(modelfile), best_modelfile)
        else:
            modelfile = args.model + '_best' + modelext
            logging.info('writing model params to ' + modelfile)
            if os.path.exists(modelfile):
                os.remove(modelfile)
            torch.save(model, modelfile)
    logging.info('----------------')
logging.info('the best model is epoch %d.' % bestmodel_num)
