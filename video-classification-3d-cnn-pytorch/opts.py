import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input', type=str, help='Input file path')
    parser.add_argument('--video_root', default='', type=str, help='Root path of input videos')
    parser.add_argument('--model', default='', type=str, help='Model file path')
    parser.add_argument('--output', default='output.json', type=str, help='Output file path')
    parser.add_argument('--temporal_output', default='output.json', type=str, help='Output file path')
    parser.add_argument('--mode', default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--model_name', default='resnet', type=str, help='Currently only support resnet')
    parser.add_argument('--model_depth', default=34, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='A', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument('--tmp', default='tmp', type=str, help='temporary folder of video images as input to CNN')
    parser.add_argument('--start_idx', default=0, type=int, help='start index of video files')
    parser.add_argument('--end_idx', default=10000, type=int, help='end index of video files')
    parser.add_argument('--sample_duration', default=16, type=int, help='sampled clip duration')
    parser.add_argument('--spatio_temporal', default=0, type=int, help='extract spatio-temporal features')
    parser.add_argument('--temporal_only', default=0, type=int, help='extract temporal-only features')
    parser.add_argument('--stride', default=-1, type=int, help='striding to sample next video clips')
    parser.set_defaults(verbose=False)
    parser.add_argument('--verbose', action='store_true', help='verbose to print out extraction process')
    parser.add_argument('--vidset', default=None, type=str, help='specific video set input if any')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return args
