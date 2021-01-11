device=$1
is_test=false
num_workers=4

#./scripts/exec.sh $device 1 $is_test "resnext_st" resnext 50 13000 0.2 128 8 3 summary 1 32 1 query,cap 1 12 \
#    0 seq none dyn 22 0 3 3 0 1 1 1 $num_workers $device
    
./scripts/exec.sh $device 1 $is_test "i3d_flow" i3d 50 13000 0.2 128 8 3 summary 1 32 1 query,cap 1 12 \
    0 seq none dyn 22 0 3 3 0 0 0 1 $num_workers $device

#./scripts/exec.sh $device 2 $is_test "resnext_st" resnext101 50 13000 0.2 128 8 3 none 1 32 1 query 1 12 \
#    0 seq none dyn 22 1 3 0 0 1 1 1 $num_workers $device
