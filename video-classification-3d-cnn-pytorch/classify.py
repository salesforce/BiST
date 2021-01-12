import torch
from torch.autograd import Variable
import pdb 

from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding

def classify_video(video_dir, video_name, class_names, model, opt):
    assert opt.mode in ['score', 'feature']

    spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterCrop(opt.sample_size),
                                 ToTensor(),
                                 Normalize(opt.mean, [1, 1, 1])])
    temporal_transform = LoopPadding(opt.sample_duration)
    data = Video(video_dir, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                 sample_duration=opt.sample_duration,
                 stride=opt.stride)
    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=opt.n_threads, pin_memory=True)

    st_outputs = []
    t_outputs = []
    video_segments = []
    for i, (inputs, segments) in enumerate(data_loader):
        with torch.no_grad():
            inputs = Variable(inputs) 
        outputs = model(inputs)
        
        st_outputs.append(outputs['spatio-temporal'].cpu().data)
        video_segments.append(segments)

    st_outputs = torch.cat(st_outputs)
    video_segments = torch.cat(video_segments)
    
    results = {
        'video': video_name,
        'st_feature': st_outputs.numpy(),
    }
    
    return results
