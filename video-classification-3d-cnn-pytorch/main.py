import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video

import glob
from tqdm import tqdm
import pdb 
import pickle as pkl

def convert_to_np(result):
    features = result['feature']
    return features
        
if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.n_classes = 400

    
    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'

    if os.path.exists(opt.tmp):
        subprocess.call('rm -rf {}'.format(opt.tmp), shell=True)
    subprocess.call('mkdir {}'.format(opt.tmp), shell=True)
        
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)

    if 'gif' in opt.video_root:
        ext = 'gif'
    else:
        ext = 'mp4'
    files = sorted(glob.glob(opt.video_root + '/*.{}'.format(ext)))
    files = files[opt.start_idx:opt.end_idx]
    
    if opt.vidset is not None:
        vidset = pkl.load(open(opt.vidset, 'rb'))
    else:
        vidset = None 
    
    for video_path in tqdm(files):
        input_file = video_path.split('/')[-1]
        video_name = input_file.split('.')[0]
        if not os.path.exists(video_path) or (vidset is not None and video_name not in vidset): 
            continue 
        subprocess.call('mkdir {}/{}'.format(opt.tmp, video_name), shell=True)
        subprocess.call('ffmpeg -loglevel quiet -nostats -i {} -vsync 0 {}/{}/image_%05d.jpg'.format(video_path, opt.tmp, video_name), shell=True)
        result = classify_video('{}/{}'.format(opt.tmp, video_name), input_file, class_names, model, opt)
        video_name = input_file.split('.')[0]
        result['st_feature'].dump(opt.output + '/' + video_name + '.npy')
        subprocess.call('rm -rf {}/{}'.format(opt.tmp, video_name), shell=True)
    
    if os.path.exists(opt.tmp):
        subprocess.call('rm -rf {}'.format(opt.tmp), shell=True)

