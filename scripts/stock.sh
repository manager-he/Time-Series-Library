export CUDA_VISIBLE_DEVICES=0

model_name=Autoformer

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./stock_data/train_set/ \
  --data_path STshimao.csv \
  --model_id ST世茂_4_1 \
  --model $model_name \
  --data stock \
  --features S \
  --target open \
  --seq_len 4 \
  --label_len 1 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./stock_data/train_set/ \
  --data_path STshimao.csv \
  --model_id ST世茂_4_1 \
  --model $model_name \
  --data stock \
  --features S \
  --target open \
  --seq_len 4 \
  --label_len 1 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./stock_data/train_set/ \
  --data_path STshimao.csv \
  --model_id ST世茂_4_4 \
  --model $model_name \
  --data stock \
  --features S \
  --target open \
  --seq_len 4 \
  --label_len 2 \
  --pred_len 4 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./stock_data/train_set/ \
  --data_path STshimao.csv \
  --model_id ST世茂_4_8 \
  --model $model_name \
  --data stock \
  --features S \
  --target open \
  --seq_len 4 \
  --label_len 2 \
  --pred_len 8 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1