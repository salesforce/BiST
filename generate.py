#!/usr/bin/env python

import argparse
import logging
import math
import sys
import time
import os
import copy
import pickle
import json
import numpy as np
import six
import pdb
from tqdm import tqdm
import torch
import torch.nn as nn
from configs.test_configs import *
from model.decode import * 
import data.data_handler as dh

# Evaluation routine
def generate_response(model, data, loader, vocab, maxlen, beam=5, penalty=2.0, nbest=1, ref_data=None):
    vocablist = sorted(vocab.keys(), key=lambda s:vocab[s])
    result_dialogs = []
    model.eval()
    with torch.no_grad():
        qa_id = 0
        it = iter(loader)
        for idx, dialog in enumerate(data['original']['dialogs']):
            vid = dialog['image_id']
            if args.undisclosed_only:
                out_dialog = dialog['dialog'][-1:]
                if ref_data is not None:
                    ref_dialog = ref_data['dialogs'][idx]
                    assert ref_dialog['image_id'] == vid 
                    ref_dialog = ref_dialog['dialog'][-1:]
            else:
                out_dialog = dialog['dialog']
            pred_dialog = {'image_id': vid,
                           'dialog': copy.deepcopy(out_dialog)}
            result_dialogs.append(pred_dialog)
            for t, qa in enumerate(out_dialog):
                if args.undisclosed_only:
                    assert qa['answer'] == '__UNDISCLOSED__'
                logging.info('%d %s_%d' % (qa_id, vid, t))
                logging.info('QS: ' + qa['question'])
                if args.undisclosed_only and ref_data is not None:
                    logging.info('REF: ' + ref_dialog[t]['answer'])
                else:
                    logging.info('REF: ' + qa['answer'])
                # prepare input data
                start_time = time.time()
                batch = next(it)
                batch.move_to_cuda()
                assert vid == batch.vids[0]
                assert qa_id == batch.qa_ids[0]
                qa_id += 1
                if args.decode_style == 'beam_search': 
                  pred_out, _ = beam_search_decode(model, batch, maxlen, start_symbol=vocab['<sos>'], unk_symbol=vocab['<unk>'], end_symbol=vocab['<eos>'], pad_symbol=vocab['<blank>'], train_args=train_args, nbest=nbest, beam=beam, dec_eos=args.dec_eos)
                  for n in range(min(nbest, len(pred_out))):
                    pred = pred_out[n]
                    hypstr = []
                    for w in pred[0]:
                        if w == vocab['<eos>']:
                            break
                        hypstr.append(vocablist[w])
                    hypstr = " ".join(hypstr)
                    logging.info('HYP[%d]: %s  ( %f )' % (n + 1, hypstr, pred[1]))
                    if n == 0: 
                        pred_dialog['dialog'][t]['answer'] = hypstr
                elif args.decode_style == 'greedy': 
                  output = greedy_decode(model, batch, maxlen, start_symbol=vocab['<sos>'], pad_symbol=vocab['<blank>'])
                  output = [i for i in output[0].cpu().numpy()]
                  hypstr = []
                  for i in output[1:]:
                    if i == vocab['<eos>']:
                        break
                    hypstr.append(vocablist[i])
                  hypstr = ' '.join(hypstr)
                  logging.info('HYP: {}'.format(hypstr))
                  pred_dialog['dialog'][t]['answer'] = hypstr
                logging.info('ElapsedTime: %f' % (time.time() - start_time))
                logging.info('-----------------------')
            #if idx == 10: break
    return {'dialogs': result_dialogs}
 
logging.info('Loading model params from ' + args.model)
path = args.model_conf
with open(path, 'rb') as f:
    vocab, train_args = pickle.load(f)
model = torch.load(args.model+'.pth.tar')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# report data summary
logging.info('#vocab = %d' % len(vocab))
# prepare test data
logging.info('Loading test data from ' + args.test_set)
skip = train_args.skip if hasattr(train_args, 'skip') else 1
train_args.fea_type = ['resnext_st']
test_data = dh.load(train_args.fea_type, args.test_path, args.test_set,
                    vocab=vocab, 
                    include_caption=train_args.include_caption, separate_caption=train_args.separate_caption,
                    max_history_length=train_args.max_history_length,
                    merge_source=train_args.merge_source,
                    undisclosed_only=args.undisclosed_only,
                    skip=skip)
test_dataloader, test_samples = dh.create_dataset(test_data, 1, False, 
                                  include_caption=train_args.include_caption, separate_caption=train_args.separate_caption,
                                  cut_a=False, num_workers=args.num_workers)

logging.info('#test sample = %d' % test_samples)
logging.info('#test batch = %d' % len(test_dataloader))

# generate sentences
logging.info('-----------------------generate--------------------------')
start_time = time.time()
labeled_test = None 
if args.undisclosed_only and args.labeled_test is not None:
    labeled_test = json.load(open(args.labeled_test, 'r'))
result = generate_response(model, test_data, test_dataloader, vocab, 
                           maxlen=args.maxlen, beam=args.beam, 
                           penalty=args.penalty, nbest=args.nbest, ref_data=labeled_test)
logging.info('----------------')
logging.info('wall time = %f' % (time.time() - start_time))
if args.output:
    logging.info('writing results to ' + args.output)
    json.dump(result, open(args.output, 'w'), indent=4)
logging.info('done')
