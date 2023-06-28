A1=0
A2=1
A3=2
A4=3
B1=4
B2=5
B3=6
B4=7

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

exp_time=$(date "+%m%d/%H%M")
log_dir="log/${exp_time}/"
mkdir -p "$log_dir"