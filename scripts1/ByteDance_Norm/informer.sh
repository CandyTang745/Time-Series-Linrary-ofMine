# export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/LongForecasting/Norm" ]; then
    mkdir ./logs/LongForecasting/Norm
fi
if [ ! -d "./logs/LongForecasting/Norm/Informer" ]; then
    mkdir ./logs/LongForecasting/Norm/Informer
fi
# seq_len=700
model_name=Informer
#iaas:93  faas:226 paas:426  rds:1113
#iaas:93  pred_len:96 144 192 336
#faas:226 pred_len:48 96 144 192
#paas:426 pred_len:96 192 336 720
#rds:1113 pred_len:96 192 336 720
enc_in=93 
GPU=0
seq_len=96
seed=512
bs=32
# CUDA_VISIBLE_DEVICES=$GPU \
# --hop_length 32 \
for data_name in iaas_wide #    
do
for seq_len in 96
do
for pred_len in 96 144 192 336 #192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset_norm/ \
    --data_path $data_name.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features MS \
    --target instance_1 \
    --freq t \
    --num_workers 0 \
    --inverse \
    --use_gpu True \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in $enc_in \
    --dec_in $enc_in \
    --c_out $enc_in \
    --des 'Exp' \
    --itr 1 \
    > logs/LongForecasting/Norm/Informer/$data_name'_'$seq_len'_'$pred_len.logs

done
done
done