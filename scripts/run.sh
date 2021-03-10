device=$1 
stage=$2
task=$3

is_test=1
num_workers=8

./scripts/exec.sh $device $stage $task $is_test $num_workers 
   
