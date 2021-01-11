device=$1
is_test=$2
num_workers=$3

#./scripts/run_new5.sh $device 1 $is_test "resnext_101_st" resnext101 30 12880 0.2 128 8 1 summary 1 32 1 query,cap 1 12 \
#    0 seq none none 22 1 1 1 0 1 1 1 $num_workers $device
 
./scripts/run_new5.sh $device 1 $is_test "resnext_101_st" resnext101 30 12880 0.2 128 8 1 summary 1 32 1 query,cap 1 12 \
    0 seq none none 22 0 1 1 0 1 1 1 $num_workers $device
    
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
nb_workers=${31}