# 当前时间
current_time=$(date "+%m%d/%H%M")

# 创建目录
output_dir="output/fusion/${current_time}"
mkdir -p "$output_dir"

# 写文件的时候，文件名使用配置文件的名字，这样就不会覆盖了

# nohup python -u fusion_main.py --gpu 6 --opt mid > ${output_dir}/mid_simple.log &
# nohup python -u fusion_main.py --gpu 6 --opt fir > ${output_dir}/fir_simple.log &
# nohup python -u fusion_main.py --gpu 5 --opt las > ${output_dir}/las_simple.log &
# nohup python -u fusion_main.py --gpu 5 --opt ran > ${output_dir}/ran_simple.log &

nohup python -u fusion_main.py --gpu 4 --opt mid --model B --pretrain > ${output_dir}/mid_pretrain.log &
nohup python -u fusion_main.py --gpu 4 --opt mid --model B > ${output_dir}/mid.log &
