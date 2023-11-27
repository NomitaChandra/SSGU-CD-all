#!/bin/sh

python train_biored.py --task biored --gnn GCN --use_gcn mentions --demo false --seed 66
sleep 100
python train_biored.py --task biored --gnn GCN --use_gcn tree --demo false --seed 66
sleep 100
python train_biored.py --task biored --gnn GCN --use_gcn both --demo false --seed 66
sleep 100
python train_biored.py --task biored --gnn GCN --use_gcn mentions --demo false --seed 68
sleep 100
python train_biored.py --task biored --gnn GCN --use_gcn tree --demo false --seed 68
sleep 100
python train_biored.py --task biored --gnn GCN --use_gcn both --demo false --seed 68
sleep 100
python train_biored.py --task biored --gnn GCN --use_gcn mentions --demo false --seed 70
sleep 100
python train_biored.py --task biored --gnn GCN --use_gcn tree --demo false --seed 70
sleep 100
python train_biored.py --task biored --gnn GCN --use_gcn both --demo false --seed 70