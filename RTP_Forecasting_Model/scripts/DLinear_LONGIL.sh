if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
seq_len=336
pred_len=24
model_name=DLinear

root_path_name=/path/to/PSCC_Code_Repo/dataset
data_path_name=LONGIL_raw.csv
model_id_name=LONGIL_raw
data_name=custom
random_seed=2021

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 321 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 5\
    --lradj 'TST'\
    --pct_start 0.2\
    --itr 1\
    --batch_size 32\
    --learning_rate 0.0001 >logs/$model_name'_'$model_id_name.log