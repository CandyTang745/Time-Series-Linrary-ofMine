#--model-hyper-params '{"low_cut": 20, "d_model": 128, "d_ff": 256,
#要指定哪张显卡跑实验时，只用保留下面这一行代码就行，不要重复在多个地方指定显卡号
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
if [ ! -d "./logs/LongForecasting/Norm/Fremer" ]; then
    mkdir ./logs/LongForecasting/Norm/Fremer
fi
# seq_len=700
model_name=Fremer
#iaas:93  faas:226 paas:426  rds:1113
enc_in=93 
# GPU=1
seq_len=96
seed=512
bs=32
#iaas:93  pred_len:96 144 192 336
#faas:226 pred_len:48 96 144 192
#paas:426 pred_len:96 192 336 720
#rds:1113 pred_len:96 192 336 720

# CUDA_VISIBLE_DEVICES=$GPU \
# --hop_length 32 \
for data_name in iaas_wide #    
do
for seq_len in 96
do
for pred_len in 96 144 192 336
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
    --pred_len $pred_len \
    --low_cut 20 \
    --d_ff 256 \
    --d_model 128 \
    --e_layers 2 \
    --enc_in $enc_in \
    --dec_in $enc_in \
    --c_out $enc_in \
    --des 'Exp' \
    --seed $seed \
    --patience 10 \
    --batch_size $bs \
    --learning_rate 0.0003 \
    --itr 1   > logs/LongForecasting/Norm/Fremer/$data_name'_'$seq_len'_'$pred_len.logs

done
done
done

