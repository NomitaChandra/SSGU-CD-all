#!/bin/sh

python train_bio.py --task cdr --gnn GCN --s0 0.1 --use_gcn tree --seed 66
sleep 100
python train_bio.py --task cdr --gnn GCN --s0 0.2 --use_gcn tree --seed 66
sleep 100
python train_bio.py --task cdr --gnn GCN --s0 0.3 --use_gcn tree --seed 66
sleep 100
python train_bio.py --task cdr --gnn GCN --s0 -0.1 --use_gcn tree --seed 66
sleep 100
python train_bio.py --task cdr --gnn GCN --s0 -0.2 --use_gcn tree --seed 66
sleep 100
python train_bio.py --task cdr --gnn GCN --s0 -0.3 --use_gcn tree --seed 66
sleep 100
#s0=0.2
#python train_bio.py --task cdr --gnn GCN --s0 s0 --use_gcn mentions --seed 66
#sleep 100
#python train_bio.py --task cdr --gnn GCN --s0 s0 --use_gcn tree --seed 66
#sleep 100
#python train_bio.py --task cdr --gnn GCN --s0 s0 --use_gcn false --seed 66
#sleep 100
#python train_bio.py --task cdr --gnn GCN --s0 s0 --use_gcn mentions --seed 68
#sleep 100
#python train_bio.py --task cdr --gnn GCN --s0 s0 --use_gcn tree--seed 68
#sleep 100
#python train_bio.py --task cdr --gnn GCN --s0 s0 --use_gcn false --seed 68
#sleep 100
#python train_bio.py --task cdr --gnn GCN --s0 s0 --use_gcn mentions --seed 70
#sleep 100
#python train_bio.py --task cdr --gnn GCN --s0 s0 --use_gcn tree --seed 70
#sleep 100
#python train_bio.py --task cdr --gnn GCN --s0 s0 --use_gcn false --seed 70