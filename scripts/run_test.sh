#!/bin/sh

python train_bio.py --task cdr --loss BSCELoss --use_gcn false --seed 66 --load_path /py_project/DGUNet-CD/out

python train_bio.py --task cdr --loss BSCELoss --use_gcn false --seed 68 --load_path /py_project/DGUNet-CD/out

python train_bio.py --task cdr --loss BSCELoss --use_gcn false --seed 70 --load_path /py_project/DGUNet-CD/out

python train_bio.py --task cdr --loss BSCELoss --use_gcn false --seed 72 --load_path /py_project/DGUNet-CD/out
