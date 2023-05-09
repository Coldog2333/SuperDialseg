export TOKENIZERS_PARALLELISM=false

python3 ../main.py \
  --dataset_name "superseg,tiage,dialseg711" \
  --test_batch_size 1 \
  --num_workers 4 \
  --model_name_or_path '../../../.cache/model_zoo/csm/csm-dailydial.pkl' \
  --model "csm"
