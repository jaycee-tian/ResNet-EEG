# 当前时间
current_time=$(date "+%m%d/%H%M")

# 创建目录
output_dir="output/${current_time}/simple"
mkdir -p "$output_dir"

# 写文件的时候，文件名使用配置文件的名字，这样就不会覆盖了

# nohup python -u main.py --gpu 0 --opt mid --simple > ${output_dir}/mid.log &
# nohup python -u main.py --gpu 0 --opt fir --simple > ${output_dir}/fir.log &
# nohup python -u main.py --gpu 0 --opt las --simple > ${output_dir}/las.log &
# nohup python -u main.py --gpu 0 --opt ran --simple > ${output_dir}/ran.log &
nohup python -u main.py --gpu 0 --opt avg --simple > ${output_dir}/avg.log &
