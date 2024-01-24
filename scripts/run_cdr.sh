#!/bin/sh

#python train_bio.py --task cdr --gnn GCN --s0 111 --use_gcn tree --seed 66
#sleep 100
#python train_bio.py --task cdr --gnn GCN --s0 111 --use_gcn tree --seed 68
#sleep 100
#python train_bio.py --task cdr --gnn GCN --s0 111 --use_gcn tree --seed 70
#sleep 100
#python train_bio.py --task cdr --gnn GCN --s0 0.4 --use_gcn tree --seed 66
#sleep 100
#python train_bio.py --task cdr --gnn GCN --s0 0.5 --use_gcn tree --seed 66
#sleep 100
#python train_bio.py --task cdr --gnn GCN --s0 0.6 --use_gcn tree --seed 66
#sleep 100
#python train_bio.py --task cdr --gnn GCN --s0 0.7 --use_gcn tree --seed 66
#sleep 100
#python train_bio.py --task cdr --gnn GCN --s0 0.8 --use_gcn tree --seed 66
#sleep 100

load_path=""
#load_path == /py_project/DGUNet-CD/out/train_filter_bert_cdr_seed_best
#losses=(BalancedLoss ATLoss AsymmetricLoss APLLoss)
losses=BSCELoss
use_gcns=(both mentions)
seeds=(66 68 70 72 74)
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