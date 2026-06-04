# export CUDA_VISIBLE_DEVICES=1
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/LongForecasting/Norm" ]; then
    mkdir ./logs/LongForecasting/Norm
fi
if [ ! -d "./logs/LongForecasting/Norm/FreqCycle" ]; then
    mkdir ./logs/LongForecasting/Norm/FreqCycle
fi
# if [ ! -d "./logs/LongForecasting/DTAF/InstanceIndependent/" ]; then
#     mkdir ./logs/LongForecasting/DTAF/InstanceIndependent/
# fi

model_name=FreqCycle
#faas:226 iaas:93 paas:426 rds:1113
#当pred_len=720时太长，验证和测试样本量不足
#iaas:93  pred_len:96 144 192 336
#faas:226 pred_len:48 96 144 192
#paas:426 pred_len:96 192 336 720
#rds:1113 pred_len:96 192 336 720
enc_in=1113
GPU=0
seed=512
bs=32
seg_window=6
seg_stride=6
#48 96 144 192
model_type='mlp'
for data_name in rds_wide
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
    --cycle 168 \
    --model_type $model_type \
    --seg_window $seg_window \
    --seg_stride $seg_stride \
    --window_type rect \
    --enc_in $enc_in \
    --dec_in $enc_in \
    --c_out $enc_in \
    --des 'Exp' \
    --seed $seed \
    --patience 10 \
    --batch_size $bs \
    --learning_rate 0.0003 \
    --itr 1 \
    > logs/LongForecasting/Norm/FreqCycle/$data_name'_'$seq_len'_'$pred_len.logs

done
done
done
#--inverse \去掉该参数可以减小评估指标的值,或者是先对数据集做预处理