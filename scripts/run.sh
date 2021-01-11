device=$1
stage=$2
is_test=true
num_workers=4

./scripts/exec.sh $device $stage $is_test 1 1 $num_workers 
