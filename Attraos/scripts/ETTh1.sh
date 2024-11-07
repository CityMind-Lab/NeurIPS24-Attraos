model_name=Attraos
for seq_len in 96
do
for layers in 1 #2
do
for pred_len in 96 192 336 720
do

python run.py \
    --model_id ETTh1_$model'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --itr 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --batch_size 32 \
    --patch_len 16 \
    --stride 16 \
    --e_layers $layers \
    --learning_rate 0.0001 \
    --is_training 1 \
    --gpu 0 \
    --FFT_evolve True \
    --multi_res True \
    

done
done
done