start=$2
end=$3
tmp=tmp_charades_test_${start}_${end}

model=resnext-101-kinetics.pth 
duration=16
stride=4
video_root=data/Charades_vu17_test_480/
output=output/resnext_st/

CUDA_VISIBLE_DEVICES=$1 python main.py --video_root $video_root \
    --output $output \
    --model $model --mode feature \
    --model_name resnext --model_depth 101 --tmp $tmp \
    --resnet_shortcut B --sample_duration $duration --n_threads 0 \
    --spatio_temporal 1 \
    --start_idx $start --end_idx $end \
    --stride $stride     
