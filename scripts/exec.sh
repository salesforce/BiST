#!/bin/bash

./scripts/path.sh

#input choices
device=$1
stage=$2       	# <=1: preparation <=2: training <=3: generating <=4: evaluating 
test_mode=$3    # true: test run with small datasets OR false: run with real datasets 
t2s=$4
s2t=$5
nb_workers=$6

# data setting 
decode_data=off                 	#use official data for testing 
undisclosed_only=1              	#only decode undisclosed dialogue turns in official data 
data_root=../../../data/dstc7/   	#TODO: replace the local data folder here 
fea_dir=$data_root
fea_file="<FeaType>/<ImageID>.npy" 
fea_type=resnext_st     # resnext_spatiotemporal
fea_names=resnext
include_caption=summary

# model setting 
d_model=128
att_h=8
nb_blocks=3
nb_venc_blocks=3
nb_cenc_blocks=3
d_ff=$(( d_model*4 ))   			# feed-forward hidden layer 

# training setting
num_epochs=50           			# e.g. 15
warmup_steps=13000      			# e.g. 9660
dropout=0.2             			# e.g. 0.1
batch_size=32
seed=1                      		# random seed 
model_prefix=bist                   # model name 
expid=t2s${t2s}_s2t${s2t}

# output folder
if [ $test_mode = true ]; then 
    expdir=exps_test/${expid}
else
    expdir=exps_test/${expid}                                          
fi
report_interval=100             # step interval to report losses during training

# generation setting 
decode_style=beam_search    	# beam search OR greedy 
penalty=1.0             		# penalty added to the score of each hypothesis
dec_eos=0
beam=5
nbest=5
model_epoch=best

echo Stage $stage Test Mode $test_mode Exp ID $expid

workdir=`pwd`
labeled_test=''
if [ $test_mode = true ]; then 
  train_set=$data_root/train_test.json
  valid_set=$data_root/valid_test.json
  test_set=$data_root/test_test.json
  labeled_test=$data_root/test_test.json
  eval_set=${labeled_test}
  undisclosed_only=0
  nb_blocks=1
  num_epochs=1
  expdir=${expdir}
else
  train_set=$data_root/train_set4DSTC7-AVSD.json
  valid_set=$data_root/valid_set4DSTC7-AVSD.json
  test_set=$data_root/test_set.json
  labeled_test=$data_root/test_set.json
  if [ $decode_data = 'off' ]; then
    test_set=$data_root/test_set4DSTC7-AVSD.json
    labeled_test=$data_root/lbl_test_set4DSTC7-AVSD.json
    eval_set=${labeled_test}
    if [ $undisclosed_only -eq 1 ]; then
        eval_set=$data_root/lbl_undiscloseonly_test_set4DSTC7-AVSD.json
    fi
  fi
fi
echo Exp Directory $expdir 

. utils/parse_options.sh || exit 1;

# directory and feature file setting
enc_psize_=`echo $enc_psize|sed "s/ /-/g"`
enc_hsize_=`echo $enc_hsize|sed "s/ /-/g"`
fea_type_=`echo $fea_type|sed "s/ /-/g"`

# command settings
train_cmd=""
test_cmd=""
gpu_id=`utils/get_available_gpu_id.sh`

set -e
set -u
set -o pipefail

# preparation
echo -------------------------
echo stage 0: preparation 
echo -------------------------
echo setup ms-coco evaluation tool
if [ ! -d utils/coco-caption ]; then
    git clone https://github.com/tylin/coco-caption utils/coco-caption
    patch -p0 -u < utils/coco-caption.patch
else
    echo Already exists COCO package.
fi

# training phase
mkdir -p $expdir
if [ $stage -eq 1 ]; then
    echo -------------------------
    echo stage 1: model training
    echo -------------------------
    CUDA_VISIBLE_DEVICES=$device python train.py \
      --gpu $gpu_id \
      --fea-type $fea_type \
      --train-path "$fea_dir/$fea_file" \
      --train-set $train_set \
      --valid-path "$fea_dir/$fea_file" \
      --valid-set $valid_set \
      --test-set $test_set \
      --num-epochs $num_epochs \
      --batch-size $batch_size \
      --model $expdir/$model_prefix \
      --rand-seed $seed \
      --report-interval $report_interval \
      --nb-blocks $nb_blocks \
      --include-caption $include_caption \
      --warmup-steps $warmup_steps \
      --nb-blocks $nb_blocks \
      --d-model $d_model \
      --d-ff $d_ff \
      --att-h $att_h \
      --dropout $dropout \
      --nb-venc-blocks $nb_venc_blocks \
      --nb-cenc-blocks $nb_cenc_blocks \
      --t2s $t2s --s2t $s2t \
      --num-workers $nb_workers \
      --device $device
fi

# testing phase
if [ $stage -eq 2 ]; then
    echo -----------------------------
    echo stage 2: generate responses
    echo -----------------------------
    if [ $decode_data = 'off' ]; then
        if [ $test_mode != true ]; then
            #fea_file="<FeaType>_testset/<ImageID>.npy"
            fea_file="<FeaType>/<ImageID>.npy"
        fi
    fi
    for data_set in $test_set; do
        echo start response generation for $data_set
        target=$(basename ${data_set%.*})
        result=${expdir}/result_${target}_ep${model_epoch}_b${beam}_p${penalty}_n${nbest}_l${maxlen}.json
        test_log=${result%.*}.log
        CUDA_VISIBLE_DEVICES=$device python generate.py \
          --gpu $gpu_id \
          --test-path "$fea_dir/$fea_file" \
          --test-set $data_set \
          --model-conf $expdir/${model_prefix}.conf \
          --model $expdir/${model_prefix}_${model_epoch} \
          --beam $beam \
          --penalty $penalty \
          --nbest $nbest \
          --output $result \
          --decode-style ${decode_style} \
          --undisclosed-only ${undisclosed_only} \
          --labeled-test ${labeled_test} \
          --maxlen ${maxlen} \
          --dec-eos ${dec_eos} 
          #--num-workers $nb_workers
         #|& tee $test_log
    done
fi

# scoring only for validation set
if [ $stage -eq 2 ]; then
    echo --------------------------
    echo stage 2: score results
    echo --------------------------
    for data_set in $eval_set; do
        echo start evaluation for $data_set
        save_target=$(basename ${test_set%.*})
        result=${expdir}/result_${save_target}_ep${model_epoch}_b${beam}_p${penalty}_n${nbest}_l${maxlen}.json
        cd ./dstc7avsd_eval/
        ./dstc7avsd_eval.sh ../spatiotemporal_transformer/$result
    done 
fi
