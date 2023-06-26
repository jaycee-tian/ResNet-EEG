# 当前时间
current_time=$(date "+%m%d/%H%M")

# define type is test
target="test"

if [ $target = "test" ];
then
    output_dir="2.output/${current_time}/test/pretrain/"
else
    output_dir="2.output/${current_time}/pretrain/"
fi
mkdir -p "$output_dir"

if [ $target = "test" ]; then
  nohup python -u seperate_main.py --gpu 5 --test > ${output_dir}/mid.log &
else
  nohup python -u seperate_main.py --gpu 5 > ${output_dir}/mid.log &
fi