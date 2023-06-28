# 当前时间
current_time=$(date "+%m%d/%H%M")

gpuA=0
gpuB=1
gpuC=2
gpuD=3
gpuE=4
gpuX=5
gpuY=6
gpuZ=7
lr="5e-4"

target="run"

model_name1="resnet18"
model_name2="resnet34"
model_name3="resnet50"
model_name4="resnet101"
model_name5="resnet152"
model_nameX0="smallnetA"
model_nameX1="smallnetB"
model_nameX2="smallnetC"
model_nameX3="smallnetD"
model_nameX4="smallnetE"
model_nameX5="smallnetF"
model_nameX6="smallnetG"
model_nameX7="smallnetH"
ptr="yes"

if [ $target = "test" ];
then
    output_dir="z-test/${current_time}/one/"
else
    output_dir="2.output/${current_time}/one/"
fi
mkdir -p "$output_dir"

if [ $target = "test" ]; then
  # nohup python -u one_main.py --gpu ${gpu} --target ${target} --ptr ${ptr} --model_name resnet18 > ${output_dir}/${model_name1}_${ptr}_mix.log &
  # nohup python -u one_main.py --gpu ${gpuA} --target ${target} --ptr ${ptr} --lr ${lr} --alpha 1.0 --model_name resnet18 > ${output_dir}/${model_name2}_${ptr}_0-5.log &
  # nohup python -u one_main.py --gpu ${gpuA} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_name1} > ${output_dir}/${model_name1}_${ptr}_1_8.log &
  nohup python -u one_main.py --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_name1} > ${output_dir}/${model_name1}_${ptr}_${lr}.log &
  
else
  # nohup python -u one_main.py --gpu ${gpuA} --target ${target} --ptr ${ptr} --lr ${lr} --model_name resnet18 > ${output_dir}/${model_name1}_${ptr}.log &
  # nohup python -u one_main.py --gpu ${gpuA} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 4 --model_name resnet34 > ${output_dir}/${model_name2}_${ptr}_1_4.log &
  # nohup python -u one_main.py --gpu ${gpuB} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name resnet34 > ${output_dir}/${model_name2}_${ptr}_1_8.log &
  # nohup python -u one_main.py --gpu ${gpuC} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 3 --grid_size 4 --model_name resnet34 > ${output_dir}/${model_name2}_${ptr}_3_4.log &
  # nohup python -u one_main.py --gpu ${gpuD} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 3 --grid_size 8 --model_name resnet34 > ${output_dir}/${model_name2}_${ptr}_3_8.log &
  # nohup python -u one_main.py --gpu ${gpuA} --target ${target} --ptr ${ptr}  --lr ${lr}  --model_name resnet50 > ${output_dir}/${model_name3}_${ptr}_mix.log &
  # nohup python -u one_main.py --gpu ${gpu} --target ${target} --ptr ${ptr} --model_name resnet101 > ${output_dir}/${model_name4}_${ptr}_mix.log &
  # nohup python -u one_main.py --gpu ${gpu} --target ${target} --ptr ${ptr} --model_name resnet152 > ${output_dir}/${model_name5}_${ptr}_mix.log &
  # ------
  # nohup python -u one_main.py --gpu ${gpuA} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_name1} > ${output_dir}/${model_name1}_${ptr}_1_8.log &
  # nohup python -u one_main.py --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_name2} > ${output_dir}/${model_name2}_${ptr}_${lr}.log &
  nohup python -u one_main.py --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_name1} > ${output_dir}/${model_name1}_${ptr}_${lr}.log &
  # nohup python -u one_main.py --gpu ${gpuC} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_name3} > ${output_dir}/${model_name3}_${ptr}_1_8.log &
  # nohup python -u one_main.py --gpu ${gpuD} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_name4} > ${output_dir}/${model_name4}_${ptr}_1_8.log &
  # nohup python -u one_main.py --gpu ${gpuE} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_name5} > ${output_dir}/${model_name5}_${ptr}_1_8.log &
#  ------
  # nohup python -u one_main.py --gpu ${gpuX} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_nameX0} > ${output_dir}/${model_nameX0}_${ptr}_1_8.log &
  
  # nohup python -u one_main.py --gpu ${gpuY} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_nameX1} > ${output_dir}/${model_nameX1}_${ptr}_1_8.log &
  
  # nohup python -u one_main.py --gpu ${gpuZ} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_nameX2} > ${output_dir}/${model_nameX2}_${ptr}_1_8.log &
  
  # nohup python -u one_main.py --gpu ${gpuA} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_nameX3} > ${output_dir}/${model_nameX3}_${ptr}_1_8.log &
  
  # nohup python -u one_main.py --gpu ${gpuB} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_nameX4} > ${output_dir}/${model_nameX4}_${ptr}_1_8.log &
  
  # nohup python -u one_main.py --gpu ${gpuC} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_nameX5} > ${output_dir}/${model_nameX5}_${ptr}_1_8.log &
  
  # nohup python -u one_main.py --gpu ${gpuD} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_nameX6} > ${output_dir}/${model_nameX6}_${ptr}_1_8.log &
  
  # nohup python -u one_main.py --gpu ${gpuE} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_nameX7} > ${output_dir}/${model_nameX7}_${ptr}_1_8.log &
  
fi