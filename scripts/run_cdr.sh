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

#losses=(BalancedLoss ATLoss AsymmetricLoss APLLoss)
losses=BSCELoss
for loss in "${losses[@]}"
do
#python train_bio.py --task cdr --gnn GCN --loss $loss --use_gcn both --seed 66
#sleep 100
#python train_bio.py --task cdr --gnn GCN --loss $loss --use_gcn mentions --seed 66
#sleep 100
python train_bio.py --task cdr --gnn GCN --loss $loss --use_gcn tree --seed 66
sleep 100
#python train_bio.py --task cdr --gnn GCN --loss $loss --use_gcn false --seed 66
#sleep 100
#python train_bio.py --task cdr --gnn GCN --loss $loss --use_gcn both --seed 68
#sleep 100
#python train_bio.py --task cdr --gnn GCN --loss $loss --use_gcn mentions --seed 68
#sleep 100
python train_bio.py --task cdr --gnn GCN --loss $loss --use_gcn tree --seed 68
sleep 100
#python train_bio.py --task cdr --gnn GCN --loss $loss --use_gcn false --seed 68
#sleep 100
#python train_bio.py --task cdr --gnn GCN --loss $loss --use_gcn both --seed 70
#sleep 100
#python train_bio.py --task cdr --gnn GCN --loss $loss --use_gcn mentions --seed 70
#sleep 100
python train_bio.py --task cdr --gnn GCN --loss $loss --use_gcn tree --seed 70
sleep 100
#python train_bio.py --task cdr --gnn GCN --loss $loss --use_gcn false --seed 70
#sleep 100

done