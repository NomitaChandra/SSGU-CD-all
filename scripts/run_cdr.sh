#!/bin/sh

losses=BSCELoss
use_gcns=("both" "mentions" "tree" "false")
for loss in "${losses[@]}"
do
  for use_gcn in "${use_gcns[@]}"
  do
    python train_bio.py --task cdr --loss $loss --use_gcn $use_gcn --seed $seed --load_path "$load_path"
    sleep 100
  done
done