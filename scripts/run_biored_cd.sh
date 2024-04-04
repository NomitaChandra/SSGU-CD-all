#!/bin/sh
export CUDA_VISIBLE_DEVICES=1
load_path=""
losses=BSCELoss
use_gcns=("tree")
seeds=(66 68 70)
rel2=(True False)
for loss in "${losses[@]}"
do
  for use_gcn in "${use_gcns[@]}"
  do
    for seed in "${seeds[@]}"
    do
      for rel in "${rel2[@]}"
      do
        python train_bio.py --task biored_cd --loss $loss --use_gcn $use_gcn --seed $seed --load_path "$load_path" --rel "$rel"
        sleep 100
      done
    done
  done
done