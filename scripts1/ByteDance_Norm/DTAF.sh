export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/LongForecasting/Norm" ]; then
    mkdir ./logs/LongForecasting/Norm
fi
if [ ! -d "./logs/LongForecasting/Norm/DTAF" ]; then
    mkdir ./logs/LongForecasting/Norm/DTAF
fi
# if [ ! -d "./logs/LongForecasting/DTAF/InstanceIndependent/" ]; then
#     mkdir ./logs/LongForecasting/DTAF/InstanceIndependent/
# fi

model_name=DTAF
#faas:226 iaas:93 paas:426 rds:1113
#当pred_len=720时太长，验证和测试样本量不足
enc_in=226
GPU=0
seed=512
bs=32

for data_name in faas_wide
do
for seq_len in 96
do
for pred_len in 96 192 336 720 
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
    --seq_len $seq_len \
    --pred_len $pred_len \
    --d_model 32 \
    --heads 4 \
    --e_layers 2 \
    --enc_in $enc_in \
    --dec_in $enc_in \
    --c_out $enc_in \
    --patch_len 16 \
    --stride 8 \
    --dropout 0.1 \
    --expert_num 2 \
    --kan_div 4 \
    --k 1 \
    --kl 1 \
    --r_dropout 1 \
    --sigma 1 \
    --aggregated_norm 1 \
    --des 'Exp' \
    --seed $seed \
    --patience 10 \
    --batch_size $bs \
    --learning_rate 0.0003 \
    --itr 1 \
    > logs/LongForecasting/Norm/DTAF/$data_name'_'$seq_len'_'$pred_len.logs

done
done
done