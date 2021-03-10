import argparse
import logging
import random
import numpy as np
import pdb 

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
# Data
parser.add_argument('--task', type=str , default='Count', help='TGIFQA task: one of [Count, Action, FrameQA, Trans]')
parser.add_argument('--test-mode', default=0, type=int, help='test mode for debugging') 
parser.add_argument('--fea-type', nargs='+', type=str, help='Image feature files (.pkl)')
parser.add_argument('--fea-path', default='', type=str,help='Path to feature files')
parser.add_argument('--merge-source', default=0, type=int, help='merge all source sequences into one') 
parser.add_argument('--model', default=None, type=str,help='output path of model and params')
parser.add_argument('--num-workers', default=0, type=int, help='data workers')
parser.add_argument('--device', default='0', type=str, help='gpu device id')

# Model 
parser.add_argument('--nb-blocks', default=6, type=int,help='number of transformer response decoder layers')
parser.add_argument('--nb-venc-blocks', default=0, type=int,help='number of transformer visual attention layers')
parser.add_argument('--d-model', default=512, type=int, help='dimension of model tensors') 
parser.add_argument('--d-ff', default=2048, type=int, help='dimension of feed forward') 
parser.add_argument('--att-h', default=8, type=int, help='number of attention heads') 
parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')  
parser.add_argument('--vid-pos', default=0, type=int, help='encode video temporal positions')
parser.add_argument('--dec-st-combine', default='seq', type=str, help='')
parser.add_argument('--enc-st-combine', default='none', type=str, help='')
parser.add_argument('--enc-vc-combine', default='dyn', type=str, help='')
parser.add_argument('--vid-enc-mode', default=22, type=int, help='version control of video encoder networks (for debugging)')
parser.add_argument('--auto-encoder', default=1, type=int, help='combine decoder with a question auto-encoder (refer to MTN)')
parser.add_argument('--t2s', default=1, type=int, help='use temporal-to-spatial reasoning direction')
parser.add_argument('--s2t', default=1, type=int, help='use spatial-to-temporal reasoning direction')
parser.add_argument('--prior', default=1, type=int, help='use a prior vector for candidate answer representations')

# Training 
parser.add_argument('--num-epochs', '-e', default=15, type=int,help='Number of epochs')
parser.add_argument('--rand-seed', '-s', default=1, type=int, help="seed for generating random numbers")
parser.add_argument('--batch-size', '-b', default=32, type=int,help='Batch size in training')
parser.add_argument('--report-interval', default=100, type=int,help='report interval to log training results')
parser.add_argument('--warmup-steps', default=4000, type=int,help='warm up steps for optimizer') 
parser.add_argument('--save-all', default=0, type=int, help='save all epoch checkpoints')

# others
parser.add_argument('--verbose', '-v', default=0, type=int,help='verbose level')

args = parser.parse_args()
args.merge_source = bool(args.merge_source)

# Presetting
random.seed(args.rand_seed)
np.random.seed(args.rand_seed)
if args.verbose >= 1:
    logging.basicConfig(level=logging.DEBUG, 
        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
else:
    logging.basicConfig(level=logging.INFO, 
        format='%(asctime)s %(levelname)s: %(message)s')
for arg in vars(args):
    print("{}={}".format(arg, getattr(args, arg)))

