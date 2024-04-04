#!/bin/sh
load_path=""
losses=BSCELoss
use_gcns=("tree")
seeds=(66 68 70)
for loss in "${losses[@]}"
do
  for use_gcn in "${use_gcns[@]}"
  do
    for seed in "${seeds[@]}"
    do
    python train_bio.py --task biored_cd --loss $loss --use_gcn $use_gcn --seed $seed --load_path "$load_path"
    sleep 100
    done
  done
done