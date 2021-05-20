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


def run_epoch(loader, epoch, model, loss_compute, eval=False):
    "Standard Training and Logging Function"
    total_tokens = 0 
    total_qtokens = 0
    total_loss = 0
    total_ae_temporal_loss = 0
    total_ae_spatial_loss = 0 
    nb_correct = 0
    it = tqdm(enumerate(loader),total=len(loader), desc="epoch {}/{}".format(epoch+1, args.num_epochs), ncols=0)
    for j, batch in it:  
        batch.move_to_cuda()
        out = model.forward(batch) 
        losses, correct_pred, _ = loss_compute(out, batch)
        total_loss += losses['out']
        total_ae_temporal_loss += losses['temporal_ae']
        total_ae_spatial_loss += losses['spatial_ae']
        total_tokens += batch.ntokens
        total_qtokens += batch.qntokens
        nb_correct += correct_pred
        if (j+1) % args.report_interval == 0 and not eval:
            print("Epoch: %d Step: %d Acc: %f Loss: %f AETemporalLoss: %f AESpatialLoss: %f" %
                    (epoch+1,j+1, 
                     1.0*correct_pred/batch.ntokens,
                     losses['out']/batch.ntokens, 
                     losses['temporal_ae']/batch.qntokens.float(), 
                     losses['spatial_ae']/batch.qntokens.float()))
            with open(train_log_path, "a") as f:
                f.write("{},{},{:e},{:e},{:e},{:e}\n".format(
                    epoch+1,j+1,
                    1.0*correct_pred/batch.ntokens,
                    losses['out']/batch.ntokens, 
                    losses['temporal_ae']/batch.qntokens.float(), 
                    losses['spatial_ae']/batch.qntokens.float()))
    out_losses={}
    out_losses['out']=total_loss/total_tokens
    out_losses['temporal_ae']=total_ae_temporal_loss/total_qtokens.float()
    out_losses['spatial_ae']=total_ae_spatial_loss//total_qtokens.float()
    out_acc = 1.0*nb_correct/(total_tokens)
    return out_losses, out_acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

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
answer_vocab_size=None 

train_dataloader, train_samples = create_dataloader(train_dataset, args.batch_size, True, args, num_workers=args.num_workers)
print("train samples {}".format(train_samples))
print('train batches {}'.format(len(train_dataloader)))
valid_dataloader, valid_samples = create_dataloader(val_dataset, args.batch_size, False, args, num_workers=args.num_workers)
print("valid samples {}".format(valid_samples))
print('valid batches {}'.format(len(valid_dataloader)))
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
    print('Answer vocab size {} All vocab size {}'.format(answer_vocab_size, voc_len))
    criterion = nn.CrossEntropyLoss(size_average=True).cuda()

ae_criterion = LabelSmoothing(size=voc_len, padding_idx=train_dataset.word2idx['<blank>'], smoothing=0.1)
ae_criterion.cuda()
    
# create_model
model = make_model(voc_len, answer_vocab_size, args, ft_sizes=[2048]) #embeddings=word_matrix)
model.cuda()

# save meta parameters
path = args.model + '.conf'
with open(path, 'wb') as f:
    pkl.dump(args, f, -1)
path2 = args.model + '_params.txt'
with open(path2, "w") as f: 
    for arg in vars(args):
        f.write("{}={}\n".format(arg, getattr(args, arg)))

logging.info('----------------')
logging.info('Start training')
logging.info('----------------')

# initialize status parameters
modelext = '.pth.tar'
max_valid_acc = -1.0e+10
min_valid_loss = 1.0e+10
bestmodel_num = 0
# save results 
trace_log_path = args.model + '_trace.csv'
with open(trace_log_path, "w") as f:
    f.write('epoch,split,acc,loss,ae_temporal_loss,ae_spatial_loss\n') 
train_log_path = args.model + '_train.csv'
with open(train_log_path, "w") as f:  
    f.write('epoch,step,acc,loss,ae_temporal_loss,ae_spatial_loss\n') 
print("Saving training results to {}".format(train_log_path))
print("Saving val results to {}".format(trace_log_path))   

# optimization 
model_opt = NoamOpt(args.d_model, 1, args.warmup_steps, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
train_loss = SimpleLossCompute(model.generator, model.ae_generator, criterion, ae_criterion,
                               opt=model_opt, args=args)
val_loss = SimpleLossCompute(model.generator, model.ae_generator, criterion, ae_criterion,
                             opt=None, args=args)
test_loss = SimpleLossCompute(model.generator, model.ae_generator, criterion, ae_criterion,
                             opt=None, args=args)
    
for epoch in range(args.num_epochs):
    # start training 
    model.train()
    train_losses, train_acc = run_epoch(train_dataloader, epoch, model, train_loss)
    logging.info("epoch: {} train acc: {} loss: {} aeTemporalLoss {} aeSpatialLoss {}".format(
        epoch+1, train_acc, train_losses['out'], train_losses['temporal_ae'], train_losses['spatial_ae']))
    
    with torch.no_grad(): 
		# test on validation data 
    	logging.info('-------validation--------')
    	model.eval()
    	valid_losses, valid_acc = run_epoch(valid_dataloader, epoch, model, val_loss, eval=True)
    	logging.info("epoch: {} valid acc: {} loss: {} aeTemporalLoss {} aeSpatialLoss {}".format(
        	epoch+1, valid_acc, valid_losses['out'], valid_losses['temporal_ae'], valid_losses['spatial_ae']))  
    
    	# test on test data 
    	logging.info('-------test--------')
    	model.eval()
    	test_losses, test_acc = run_epoch(test_dataloader, epoch, model, test_loss, eval=True)
    	logging.info("epoch: {} test acc: {} loss: {} aeTemporalLoss {} aeSpatialLoss {}".format(
        	epoch+1, test_acc, test_losses['out'], test_losses['temporal_ae'], test_losses['spatial_ae']))  
    
    with open(trace_log_path,"a") as f:
        f.write("{},train,{:e},{:e},{:e},{:e}\n".format(
            epoch+1, train_acc, train_losses['out'], train_losses['temporal_ae'], train_losses['spatial_ae']))
        f.write("{},val,{:e},{:e},{:e},{:e}\n".format(
            epoch+1, valid_acc, valid_losses['out'], valid_losses['temporal_ae'], valid_losses['spatial_ae']))  
        f.write("{},test,{:e},{:e},{:e},{:e}\n".format(
            epoch+1, test_acc, test_losses['out'], test_losses['temporal_ae'], test_losses['spatial_ae'])) 
    
    # update the model and save checkpoints
    valid_loss = valid_losses['out']
    if min_valid_loss > valid_loss:
        bestmodel_num = epoch+1
        logging.info('validation acc changes %.4f -> %.4f' % (max_valid_acc, valid_acc))
        logging.info('validation loss reduces %.4f -> %.4f' % (min_valid_loss, valid_loss))
        max_valid_acc = valid_acc
        min_valid_loss = valid_loss
        modelfile = args.model + '_best' + modelext
        logging.info('writing model params to ' + modelfile)
        torch.save(model, modelfile)
    logging.info('----------------')
    
logging.info('the best model is epoch %d.' % bestmodel_num)

                
