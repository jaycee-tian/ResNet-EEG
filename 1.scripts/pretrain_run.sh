# 当前时间
current_time=$(date "+%m%d/%H%M")
# current second
second=$(date "+%S")
model_name="resnet18"

batch_size=512
# lr="5e-5"

output_dir="2.output/${current_time}/beta/"

mkdir -p "$output_dir"

nohup python -u pretrain.py --model_name ${model_name} --batch_size ${batch_size} > ${output_dir}/${model_name}_${second}.log &