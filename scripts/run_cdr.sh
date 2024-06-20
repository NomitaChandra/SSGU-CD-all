#!/bin/sh

load_path=""
#load_path == /py_project/SSGU-CD/out/train_filter_bert_cdr_seed_best
#losses=(BalancedLoss ATLoss AsymmetricLoss APLLoss)
losses=BSCELoss
use_gcns=("both" "mentions" "tree" "false")
seeds=(64 66 68 70 72)
for loss in "${losses[@]}"
do
  for use_gcn in "${use_gcns[@]}"
  do
    for seed in "${seeds[@]}"
    do
    python train_bio.py --task cdr --loss $loss --use_gcn $use_gcn --seed $seed --load_path "$load_path"
    sleep 100
    done
  done
done