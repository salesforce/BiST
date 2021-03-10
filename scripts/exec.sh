#!/bin/bash

#input choices
device=$1
stage=$2       # <=1: preparation <=2: training <=3: generating <=4: evaluating 
task=$3
test_mode=$4    # true: test run with small datasets OR false: run with real datasets 
nb_workers=$5

# data setting 
fea_dir="video-classification-3d-cnn-pytorch/output/"
fea_file="<FeaType>/<ImageID>.npy" 
fea_type=resnext_st
fea_names=resnext 

# training setting 
num_epochs=50
warmup_steps=5000
dropout=0.2
batch_size=32
seed=1
model_prefix=mtn
expid=${task}_${fea_names}
report_interval=100

# model setting 
d_model=128
att_h=8
nb_blocks=3
nb_venc_blocks=3
t2s=1
s2t=1
d_ff=$(( d_model*4 ))
prior=1

# generation setting
model_epoch=best

# output folder name
if [ $test_mode -eq 1 ]; then 
    expdir=exps_test/${expid}
else
    expdir=exps/${expid}                                          
fi

echo Stage $stage Test Mode $test_mode Exp ID $expid
echo Exp Directory $expdir 

# training phase
mkdir -p $expdir
if [ $stage -eq 1 ]; then
    echo -------------------------
    echo stage 1: model training
    echo -------------------------
    CUDA_VISIBLE_DEVICES=$device python train.py \
      --test-mode $test_mode \
      --fea-type $fea_type \
      --fea-path "$fea_dir/$fea_file" \
      --num-epochs $num_epochs \
      --batch-size $batch_size \
      --model $expdir/$model_prefix \
      --rand-seed $seed \
      --report-interval $report_interval \
      --nb-blocks $nb_blocks \
      --warmup-steps $warmup_steps \
      --nb-blocks $nb_blocks \
      --d-model $d_model \
      --d-ff $d_ff \
      --att-h $att_h \
      --dropout $dropout \
      --nb-venc-blocks $nb_venc_blocks \
      --t2s $t2s --s2t $s2t \
      --num-workers $nb_workers \
      --device $device \
      --task $task
fi

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
