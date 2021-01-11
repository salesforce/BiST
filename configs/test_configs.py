import argparse
import logging
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--test-path', default='', type=str,
                    help='Path to test feature files')
parser.add_argument('--test-set', default='', type=str,
                    help='Filename of test data')
parser.add_argument('--model-conf', default='', type=str,
                    help='Attention model to be output')
parser.add_argument('--model', '-m', default='', type=str,
                    help='Attention model to be output')
parser.add_argument('--maxlen', default=12, type=int,
                    help='Max-length of output sequence')
parser.add_argument('--dec-eos', default=0, type=int, 
                    help='if zero, include eos token as skip token during decoding')
parser.add_argument('--beam', default=3, type=int,
                    help='Beam width')
parser.add_argument('--penalty', default=2.0, type=float,
                    help='Insertion penalty')
parser.add_argument('--nbest', default=5, type=int,
                    help='Number of n-best hypotheses')
parser.add_argument('--output', '-o', default='', type=str,
                    help='Output generated responses in a json file')
parser.add_argument('--verbose', '-v', default=0, type=int,
                    help='verbose level')
parser.add_argument('--decode-style', default='greedy', type=str, help='greedy or beam_search')
parser.add_argument('--undisclosed-only', default=0, type=int, help='decode undisclosed dialogue turn in test set only')
parser.add_argument('--labeled-test', default=None, type=str, help='directory to labelled data')
parser.add_argument('--num-workers', default=0, type=int, help='data workers')


args = parser.parse_args()
args.undisclosed_only = bool(args.undisclosed_only)
for arg in vars(args):
    print("{}={}".format(arg, getattr(args, arg)))

if args.verbose >= 1:
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
else:
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s')
