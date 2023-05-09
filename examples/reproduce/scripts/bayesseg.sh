#####
# Before evaluating BayesSeg, you need to build the source codes.
# You should install `ant` and then use it to build following the instruction in bayesseg/README.md
# 1. `mkdir bayesseg/classes`
# 2. `cd bayesseg` and run `ant build`
# 3. make sure `chmod 777 "./segment"`
#####

cd "../../../" || exit

cd "models/bayesseg" || exit

chmod 777 "./segment"

python3 ../../examples/reproduce/main.py \
  --dataset_name "superseg,tiage,dialseg711" \
  --test_batch_size 1 \
  --num_workers 8 \
  --cache_dir "../../.cache/bayesseg/seg-input" \
  --model "bayesseg"
