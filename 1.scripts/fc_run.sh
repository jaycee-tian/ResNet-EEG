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
    output_dir="z-test/${current_time}/fc/"
else
    output_dir="2.output/${current_time}/fc/"
fi
mkdir -p "$output_dir"

if [ $target = "test" ]; then
  # nohup python -u one_main.py --gpu ${gpu} --target ${target} --ptr ${ptr} --model_name resnet18 > ${output_dir}/${model_name1}_${ptr}_mix.log &
  # nohup python -u one_main.py --gpu ${gpuA} --target ${target} --ptr ${ptr} --lr ${lr} --alpha 1.0 --model_name resnet18 > ${output_dir}/${model_name2}_${ptr}_0-5.log &
  # nohup python -u one_main.py --gpu ${gpuA} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_name1} > ${output_dir}/${model_name1}_${ptr}_1_8.log &
  nohup python -u 0.beta/test_fc.py --gpu ${gpuA} --target ${target} --ptr ${ptr} --lr ${lr} --n_channels 1 --grid_size 8 --model_name ${model_name1} > ${output_dir}/${model_name1}_${ptr}_${lr}.log &
  
else
  nohup python -u 0.beta/test_fc.py > ${output_dir}/fc.log &
fi