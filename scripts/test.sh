#!/bin/bash

#input choices
device=$1
task=$2
test_mode=$3    # true: test run with small datasets OR false: run with real datasets 
nb_workers=$4
expdir=$5

stage=2
model_prefix=mtn

# data setting 
fea_dir="video-classification-3d-cnn-pytorch/output/"
fea_file="<FeaType>/<ImageID>.npy" 
fea_type=resnext_st
fea_names=resnext 

# generation setting
batch_size=32
model_epoch=best

# output folder name
#if [ $test_mode -eq 1 ]; then 
#    expdir=exps_archives/${expid}
#else
#    expdir=exps_archives/${expid}                                          
#fi

echo Stage $stage Test Mode $test_mode Exp ID $expid
echo Exp Directory $expdir 

if [ $stage -eq 2 ]; then
    echo -------------------------
    echo stage 2: model testing
    echo -------------------------
    CUDA_VISIBLE_DEVICES=$device python test.py \
      --test-mode $test_mode \
      --fea-type $fea_type \
      --fea-path "$fea_dir/$fea_file" \
      --batch-size $batch_size \
      --model $expdir/$model_prefix \
      --num-workers $nb_workers \
      --device $device \
      --task $task 
fi
