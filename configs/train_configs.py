import argparse
import logging
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
# Data
parser.add_argument('--fea-type', nargs='+', type=str, help='Image feature files (.pkl)')
parser.add_argument('--train-path', default='', type=str,help='Path to training feature files')
parser.add_argument('--train-set', default='', type=str,help='Filename of train data')
parser.add_argument('--valid-path', default='', type=str,help='Path to validation feature files')
parser.add_argument('--valid-set', default='', type=str,help='Filename of validation data')
parser.add_argument('--test-set', default='', type=str,help='Filename of validation data')
parser.add_argument('--include-caption', default='none', type=str, help='Include caption in the history')
parser.add_argument('--separate-caption', default=1, type=int, help='Separate caption from dialogue history')
parser.add_argument('--cut-a', default=1, type=int, help='randomly cut responses to simulate bs') 
parser.add_argument('--merge-source', default=0, type=int, help='merge all source sequences into one') 
parser.add_argument('--exclude-video', action='store_true',help='')
parser.add_argument('--word-emb', default='none', type=str, help='')
parser.add_argument('--fixed-word-emb', default=0, type=int, help='')
parser.add_argument('--model', default=None, type=str,help='output path of model and params')
parser.add_argument('--relative-pe', default=False, type=int, help='')
parser.add_argument('--eval-metric', default='loss', type=str, help='')
parser.add_argument('--cutoff', default=5, type=int, help='minimum word frequency to create vocabulary') 
parser.add_argument('--skip', default=1, type=int, help='')
parser.add_argument('--num-workers', default=0, type=int, help='')
parser.add_argument('--device', default='0', type=str, help='')

# Model 
parser.add_argument('--nb-blocks', default=6, type=int,help='number of transformer blocks')
parser.add_argument('--nb-venc-blocks', default=0, type=int,help='number of transformer blocks')
parser.add_argument('--nb-cenc-blocks', default=0, type=int, help='')
parser.add_argument('--nb-aenc-blocks', default=0, type=int, help='')
parser.add_argument('--d-model', default=512, type=int, help='dimension of model tensors') 
parser.add_argument('--d-ff', default=2048, type=int, help='dimension of feed forward') 
parser.add_argument('--att-h', default=8, type=int, help='number of attention heads') 
parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')  
parser.add_argument('--separate-his-embed', default=0, type=int, help='Separate the dialog history embedding?')
parser.add_argument('--separate-cap-embed', default=0, type=int, help='Separate the video caption embedding') 
parser.add_argument('--separate-out-embed', default=0, type=int, help='')
parser.add_argument('--separate-out-linear', default=0, type=int, help='')
parser.add_argument('--diff-encoder', default=1, type=int, help='use different encoder for the autoencoder?') 
parser.add_argument('--diff-embed', default=0, type=int, help='use different embedding for the autoencoder?') 
parser.add_argument('--diff-gen', default=0, type=int, help='use different generator for the autoencoder?') 
parser.add_argument('--auto-encoder-ft', default='query', type=str, help='use what features for autoencoder?')
parser.add_argument('--ptr-gen', default=1, type=int, help='')
parser.add_argument('--ptr-ft', default='query,cap', type=str, help='')
parser.add_argument('--mask-unk', default=1, type=int, help='')
parser.add_argument('--vid-pos', default=0, type=int, help='')
parser.add_argument('--dec-st-combine', default='seq', type=str, help='')
parser.add_argument('--enc-st-combine', default='none', type=str, help='')
parser.add_argument('--enc-vc-combine', default='dyn', type=str, help='')
parser.add_argument('--vid-enc-mode', default=22, type=int, help='')
parser.add_argument('--auto-encoder', default=0, type=int, help='')
parser.add_argument('--skip-connect', default=0, type=int, help='')
parser.add_argument('--no-ff', default=0, type=int, help='')
parser.add_argument('--dynamic-fusion', default=0, type=int, help='')
parser.add_argument('--cta', default=1, type=int, help='')
parser.add_argument('--csa', default=1, type=int, help='')
parser.add_argument('--qta', default=1, type=int, help='')
parser.add_argument('--qsa', default=1, type=int, help='')
parser.add_argument('--query-mm', default='query', type=str, help='')
parser.add_argument('--t2s', default=1, type=int, help='')
parser.add_argument('--s2t', default=1, type=int, help='')
parser.add_argument('--noW-venc', default=0, type=int, help='')

# Training 
parser.add_argument('--num-epochs', '-e', default=15, type=int,help='Number of epochs')
parser.add_argument('--rand-seed', '-s', default=1, type=int, help="seed for generating random numbers")
parser.add_argument('--batch-size', '-b', default=32, type=int,help='Batch size in training')
parser.add_argument('--max-length', default=256, type=int,help='Maximum length for controling batch size')
parser.add_argument('--max-history-length', default=-1, type=int, help='Maximum past history length to consider')
parser.add_argument('--report-interval', default=100, type=int,help='report interval to log training results')
parser.add_argument('--warmup-steps', default=4000, type=int,help='warm up steps for optimizer') 
parser.add_argument('--loss-l', default=1.0, type=float, help="")
parser.add_argument('--save-all', default=0, type=int, help='')
# others
parser.add_argument('--verbose', '-v', default=0, type=int,help='verbose level')

args = parser.parse_args()
args.separate_his_embed = bool(args.separate_his_embed)
args.separate_caption = bool(args.separate_caption)
args.merge_source = bool(args.merge_source)
args.separate_cap_embed = bool(args.separate_cap_embed)
args.cut_a = bool(args.cut_a)
args.diff_encoder = bool(args.diff_encoder)
args.diff_embed = bool(args.diff_embed)
args.diff_gen = bool(args.diff_gen)
args.fixed_word_emb = bool(args.fixed_word_emb)
args.relative_pe = bool(args.relative_pe)

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
