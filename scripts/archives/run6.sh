device=$1
is_test=$2
num_workers=$3

./scripts/run_new5.sh $device 1 $is_test "resnext_101_st" resnext101 50 13000 0.2 128 8 1 summary 1 32 1 query,cap 1 12 \
    0 seq none dyn 22 1 1 1 0 1 1 1 $num_workers $device
    
./scripts/run_new5.sh $device 2 $is_test "resnext_101_st" resnext101 50 13000 0.2 128 8 1 summary 1 32 1 query,cap 1 12 \
    0 seq none dyn 22 1 1 1 0 1 1 1 $num_workers $device