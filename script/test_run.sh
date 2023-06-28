# 当前时间
current_time=$(date "+%m%d/%H%M")


target="test"

if [ $target = "test" ];
then
    output_dir="2.output/test/${current_time}/one/"
else
    output_dir="2.output/${current_time}/one/"
fi
mkdir -p "$output_dir"

if [ $target = "test" ]; then
  nohup python -u one_main.py --gpu 7 --test > ${output_dir}/mid.log &
else
  nohup python -u one_main.py --gpu 7  > ${output_dir}/mid.log &
fi