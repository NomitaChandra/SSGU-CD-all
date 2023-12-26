#!/bin/sh

python train_bio.py --task cdr --gnn GCN --loss BalancedLoss --use_gcn tree --seed 68