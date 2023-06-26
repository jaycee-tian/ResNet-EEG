# 当前时间
current_time=$(date "+%m%d/%H%M")

# 创建目录
output_dir="output/${current_time}"
mkdir -p "$output_dir"

# 写文件的时候，文件名使用配置文件的名字，这样就不会覆盖了

# nohup python -u main.py --gpu 1 --opt mid  > ${output_dir}/mid.log &
# nohup python -u main.py --gpu 1 --opt fir  > ${output_dir}/fir.log &
# nohup python -u main.py --gpu 1 --opt las  > ${output_dir}/las.log &
# nohup python -u main.py --gpu 1 --opt ran  > ${output_dir}/ran.log &
nohup python -u main.py --gpu 0 --diff > ${output_dir}/diff_mid.log &
nohup python -u main.py --gpu 0 > ${output_dir}/mid.log &
