#!/bin/sh

#python train_bio.py --task gda --gnn GCN --use_gcn mentions --demo false --seed 66
#sleep 100
#python train_bio.py --task gda --gnn GCN --use_gcn tree --demo false --seed 66
#sleep 100
#python train_bio.py --task gda --gnn GCN --use_gcn both --demo false --seed 66
#sleep 100
python train_bio.py --task gda --gnn GCN --use_gcn false --demo false --seed 66
#sleep 100
#python train_bio.py --task gda --gnn GCN --use_gcn mentions --demo false --seed 68
#sleep 100
#python train_bio.py --task gda --gnn GCN --use_gcn tree --demo false --seed 68
#sleep 100
#python train_bio.py --task gda --gnn GCN --use_gcn both --demo false --seed 68
sleep 100
python train_bio.py --task gda --gnn GCN --use_gcn false --demo false --seed 68
#sleep 100
#python train_bio.py --task gda --gnn GCN --use_gcn mentions --demo false --seed 70
#sleep 100
#python train_bio.py --task gda --gnn GCN --use_gcn tree --demo false --seed 70
#sleep 100
#python train_bio.py --task gda --gnn GCN --use_gcn both --demo false --seed 70
sleep 100
python train_bio.py --task gda --gnn GCN --use_gcn false --demo false --seed 70