#!/bin/bash

./scripts/path.sh

#input choices
device=$1
stage=$2       # <=1: preparation <=2: training <=3: generating <=4: evaluating 
test_mode=$3    # true: test run with small datasets OR false: run with real datasets 
fea_type=$4    # "vggish" OR "i3d_flow" OR "vggish i3d_flow"
fea_names=$5    # vggish OR i3dflow OR vggish+i3dflow 
num_epochs=$6   # e.g. 15 
warmup_steps=$7 # e.g. 9660
dropout=$8      # e.g. 0.1
d_model=$9
att_h=${10}
nb_blocks=${11}
include_caption=${12}
sep_caption=${13}
batch_size=${14}
ptr=${15}
ptr_ft=${16}
mask_unk=${17}
maxlen=${18}
vpos=${19}
dec_st_combine=${20}
enc_st_combine=${21}
enc_vc_combine=${22}
vid_enc_mode=${23}
auto_encoder=${24}
nb_venc_blocks=${25}
nb_cenc_blocks=${26}
nb_aenc_blocks=${27}
t2s=${28}
s2t=${29}
cut_a=${30}
word_emb=${31}
fixed_word_emb=${32}
cutoff=${33}
nb_workers=${34}

query_mm=query
sep_out_emb=0
sep_out_linear=0
skip=1
dec_eos=0
beam=5
nbest=5
model_epoch=best

# data setting 
#batch_size=32                   # number of dialogue instances in each batch 
max_length=256                  # batch size is reduced if len(input_feature) >= max_length
#include_caption=caption,summary # concatenate caption and summary together 
#sep_caption=1                   # separate caption from history 
max_his_len=-1                  #-1 1 2 ... 10; -1 for all dialogue turns possible 
merge_source=0                  #concatenate history(+caption) and query together as one single source sequence
decode_data=off                 #use official data for testing 
undisclosed_only=1              #only decode undisclosed dialogue turns in official data 
data_root=../../data/OfficialData/   #TODO: replace the local data folder here 
fea_dir=$data_root
fea_file="<FeaType>/<ImageID>.npy" 

# model setting 
sep_his_embed=0         # separate history embedding from source sequence embedding 
sep_cap_embed=0         # separate caption embedding from source sequence embedding 
d_ff=$(( d_model*4 ))   # feed-forward hidden layer 
# auto-encoder setting  
diff_encoder=1          # use different query encoder weights in auto-encoder   
diff_embed=0            # use different query embedding weights in auto-encoder
diff_gen=0              # use different generator in auto-encoder 
auto_encoder_ft=query   # features to be auto-encoded e.g. query, caption, summary  

# training setting
#cut_a=1                     # 0: none OR 1: randomly truncated responses for token-level decoding simulation in training 
loss_l=1                    # lambda in loss function 
seed=1                      # random seed 
model_prefix=mtn                                                # model name 
    expid=mode${vid_enc_mode}_${fea_names}_cap${include_caption}_cuta${cut_a}_vpos${vpos}_bs${batch_size}_wu${warmup_steps}_eps${num_epochs}_dr${dropout}_d${d_model}_att${att_h}_Nvenc${nb_venc_blocks}_t2s${t2s}_s2t${s2t}_Ncenc${nb_cenc_blocks}_Naenc${nb_aenc_blocks}_Ndec${nb_blocks}_stenc${enc_st_combine}_vcenc${enc_vc_combine}_stdec${dec_st_combine}_ptr${ptr}_ptrft${ptr_ft}_maskunk${mask_unk}_ae${auto_encoder}_emb${word_emb}_efixed${fixed_word_emb}_cutoff${cutoff}

# output folder name
if [ $test_mode = true ]; then 
    expdir=exps_test/${expid}
else
    expdir=exps_2/${expid}                                          
fi

# generation setting 
decode_style=beam_search    # beam search OR greedy 
penalty=1.0             # penalty added to the score of each hypothesis
report_interval=100     # step interval to report losses during training 

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
echo stage 1: preparation 
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
    echo stage 2: model training
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
      --max-length $max_length \
      --model $expdir/$model_prefix \
      --rand-seed $seed \
      --report-interval $report_interval \
      --nb-blocks $nb_blocks \
      --include-caption $include_caption \
      --max-history-length $max_his_len \
      --separate-his-embed $sep_his_embed \
      --separate-caption $sep_caption \
      --merge-source $merge_source \
      --separate-cap-embed $sep_cap_embed \
      --warmup-steps $warmup_steps \
      --nb-blocks $nb_blocks \
      --d-model $d_model \
      --d-ff $d_ff \
      --att-h $att_h \
      --dropout $dropout \
      --cut-a $cut_a \
      --loss-l ${loss_l} \
      --diff-encoder ${diff_encoder} \
      --diff-embed ${diff_embed} \
      --auto-encoder-ft ${auto_encoder_ft} \
      --diff-gen ${diff_gen}  \
      --ptr-gen ${ptr} \
      --separate-out-embed ${sep_out_emb} \
      --separate-out-linear ${sep_out_linear} \
      --ptr-ft ${ptr_ft} \
      --cutoff ${cutoff} \
      --skip $skip \
      --mask-unk $mask_unk \
      --vid-pos $vpos \
      --dec-st-combine $dec_st_combine \
      --enc-st-combine $enc_st_combine \
      --enc-vc-combine $enc_vc_combine \
      --vid-enc-mode $vid_enc_mode \
      --auto-encoder $auto_encoder \
      --query-mm $query_mm \
      --nb-venc-blocks $nb_venc_blocks \
      --nb-cenc-blocks $nb_cenc_blocks \
      --nb-aenc-blocks $nb_aenc_blocks \
      --t2s $t2s --s2t $s2t \
      --word-emb $word_emb \
      --fixed-word-emb $fixed_word_emb \
      --num-workers $nb_workers \
      --device $device
fi

# testing phase
if [ $stage -eq 2 ]; then
    echo -----------------------------
    echo stage 3: generate responses
    echo -----------------------------
    if [ $decode_data = 'off' ]; then
        if [ $test_mode != true ]; then
            fea_file="<FeaType>_testset/<ImageID>.npy"
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
    echo stage 4: score results
    echo --------------------------
    for data_set in $eval_set; do
        echo start evaluation for $data_set
        save_target=$(basename ${test_set%.*})
        result=${expdir}/result_${save_target}_ep${model_epoch}_b${beam}_p${penalty}_n${nbest}_l${maxlen}.json
        cd ../dstc7avsd_eval/
        ./dstc7avsd_eval.sh ../spatiotemporal_transformer/$result
    done 
fi
