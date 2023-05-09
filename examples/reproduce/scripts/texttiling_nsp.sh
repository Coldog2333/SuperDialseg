export TOKENIZERS_PARALLELISM=false

python3 ../main.py \
  --dataset_name "superseg,tiage,dialseg711" \
  --test_batch_size 1 \
  --num_workers 4 \
  --max_utterance_len 64 \
  --cut_rate 0.5 \
  --model "texttiling_nsp"
