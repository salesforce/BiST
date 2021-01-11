device=$1
is_test=$2
num_workers=$3

./scripts/run_new7.sh $device 1 $is_test "resnext_101_st vggish" resnext101+vggish 50 13000 0.2 128 8 3 none 0 32 1 query 1 12 \
    0 seq none dyn 22 1 3 0 3 1 1 1 1 $num_workers $device
    
./scripts/run_new7.sh $device 2 $is_test "resnext_101_st vggish" resnext101+vggish 50 13000 0.2 128 8 3 none 0 32 1 query 1 12 \
    0 seq none dyn 22 1 3 0 3 1 1 1 1 $num_workers $device