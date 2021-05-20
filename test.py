import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from make_tgif_st import *
from util import AverageMeter
import pdb 
from configs.configs import *
from model.mtn import *
from model.optimize import *
from model.label_smoothing import * 
from util import AverageMeter
import pickle 
import json 

def run_epoch(loader, model, loss_compute):
    losses = AverageMeter()
    accuracy = AverageMeter()
    predictions = []
    labels = []
    qids = []
    it = tqdm(enumerate(loader),total=len(loader), ncols=0)
    for j, batch in it:  
        batch.move_to_cuda()
        out = model.forward(batch) 
        loss, acc, preds = loss_compute(out, batch, keep_average=True)
        losses.update(loss['out'], len(batch.qa_ids))
        accuracy.update(acc, len(batch.qa_ids))
        predictions.append(preds)
        labels.append(batch.trg)
        qids.extend(batch.qa_ids)
    print('loss {}, acc {}'.format(losses.avg, accuracy.avg))
    with open(args.model + '.eval', 'w') as f:
        f.write('loss,acc\n')
        f.write('{},{}'.format(losses.avg, accuracy.avg))
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    save = {}
    for idx, qid in enumerate(qids):
        save[qid] = {}
        save[qid]['prediction'] = predictions[idx].item()
        save[qid]['label'] = labels[idx].item()
    json.dump(save, open(args.model + '.pred', 'w'), indent=4)
        
        
### add arguments ###
args.vc_dir = './data/Vocabulary'
args.df_dir = './data/dataset'
args.max_sequence_length = 100

print('Start loading TGIF dataset')
train_dataset = DatasetTGIF(dataset_name='train',
                     fea_type=args.fea_type, fea_path=args.fea_path,
                     data_type=args.task,
                     dataframe_dir=args.df_dir,
                     vocab_dir=args.vc_dir,
                     is_test=args.test_mode)
train_dataset.load_word_vocabulary()

val_dataset = train_dataset.split_dataset(ratio=0.1)
val_dataset.share_word_vocabulary_from(train_dataset)

test_dataset = DatasetTGIF(dataset_name='test',
                            fea_type=args.fea_type, fea_path=args.fea_path,
                            data_type=args.task,
                            dataframe_dir=args.df_dir,
                            vocab_dir=args.vc_dir,
                            is_test=args.test_mode)
test_dataset.share_word_vocabulary_from(train_dataset)

print('dataset lengths train/val/test %d/%d/%d' % (len(train_dataset),len(val_dataset),len(test_dataset)))

voc_len = train_dataset.n_words
test_dataloader, test_samples = create_dataloader(test_dataset, args.batch_size, False, args, num_workers=args.num_workers)
print("test samples {}".format(test_samples))
print('test batches {}'.format(len(test_dataloader)))

if args.task=='Count':
    # add L2 loss
    criterion = nn.MSELoss(size_average=True).cuda()
elif args.task in ['Action','Trans']:
    from model.embed_loss import MultipleChoiceLoss
    criterion = MultipleChoiceLoss(num_option=5, margin=1, size_average=True).cuda()
elif args.task=='FrameQA':
    # add classification loss
    answer_vocab_size = len(train_dataset.ans2idx)
    print('Vocabulary size', answer_vocab_size, voc_len)
    criterion = nn.CrossEntropyLoss(size_average=True).cuda()

ae_criterion = LabelSmoothing(size=voc_len, padding_idx=train_dataset.word2idx['<blank>'], smoothing=0.1)
ae_criterion.cuda()

# load model 
logging.info('Loading model params from ' + args.model)
with open(args.model + '.conf', 'rb') as f:
    train_args = pickle.load(f)
model = torch.load(args.model + '_best.pth.tar')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

logging.info('----------------')
logging.info('Start testing')
logging.info('----------------')

test_loss = SimpleLossCompute(model.generator, model.ae_generator, criterion, ae_criterion,
                             opt=None, args=args)

with torch.no_grad():
	model.eval()
	run_epoch(test_dataloader, model, test_loss)


