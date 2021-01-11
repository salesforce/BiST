device=$1
is_test=true
num_workers=4
stage=2 

./scripts/exec.sh $device $stage $is_test 1 1 $num_workers 
