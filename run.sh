# 当前时间
# 获得gpu的编号变量
. script/prepare.sh

nohup python -u main.py --pretrain --model_name ${model_name1} > ${log_dir}/${model_name1}.log &
sleep 5
nohup python -u main.py --pretrain --model_name ${model_name2} > ${log_dir}/${model_name2}.log &
sleep 5
nohup python -u main.py --pretrain --model_name ${model_name3} > ${log_dir}/${model_name3}.log &
sleep 5
nohup python -u main.py --pretrain --model_name ${model_name4} > ${log_dir}/${model_name4}.log &
sleep 5
nohup python -u main.py --pretrain --model_name ${model_name5} > ${log_dir}/${model_name5}.log &
