#!/bin/bash

./scripts/path.sh

#input choices
device=$1
expdir=$2

stage=2
test_mode=false

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
model_prefix=mtn                  # model name 

# generation setting 
decode_style=beam_search    	# beam search OR greedy 
penalty=1.0             		# penalty added to the score of each hypothesis
beam=5
nbest=5
model_epoch=best

echo Stage $stage Test Mode $test_mode Exp ID $expid

workdir=`pwd`
labeled_test=''
if [ $test_mode = true ]; then 
  test_set=$data_root/test_test.json
  labeled_test=$data_root/test_test.json
  eval_set=${labeled_test}
  undisclosed_only=0
  nb_blocks=1
  num_epochs=1
  expdir=${expdir}
else
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

. scripts/parse_options.sh || exit 1;

# directory and feature file setting
enc_psize_=`echo $enc_psize|sed "s/ /-/g"`
enc_hsize_=`echo $enc_hsize|sed "s/ /-/g"`
fea_type_=`echo $fea_type|sed "s/ /-/g"`

# command settings
set -e
set -u
set -o pipefail

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
        result=${expdir}/result_${target}_ep${model_epoch}_b${beam}_p${penalty}_n${nbest}.json
        test_log=${result%.*}.log
        CUDA_VISIBLE_DEVICES=$device python generate.py \
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
        result=${expdir}/result_${save_target}_ep${model_epoch}_b${beam}_p${penalty}_n${nbest}.json
        cd ./dstc7avsd_eval/
        ./dstc7avsd_eval.sh ../$result
    done 
fi
