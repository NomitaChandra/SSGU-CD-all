#!/bin/bash

#losses=(BalancedLoss ATLoss AsymmetricLoss APLLoss BSCELoss)
#use_gcns=("both" "mentions" "tree" "false")
for ((i=1;i<=5;i++))
do
  python train_bio.py --task cdr --loss "BSCELoss" --use_gcn "both" --dropout 0.5 --s0 0.1
  sleep 10
done

for ((i=1;i<=5;i++))
do
  python train_bio.py --task cdr --loss "BSCELoss" --use_gcn "both" --dropout 0.5 --s0 0.2
  sleep 10
done

for ((i=1;i<=5;i++))
do
  python train_bio.py --task cdr --loss "BSCELoss" --use_gcn "both" --dropout 0.5 --s0 0.4
  sleep 10
done

for ((i=1;i<=5;i++))
do
  python train_bio.py --task cdr --loss "BSCELoss" --use_gcn "both" --dropout 0.5 --s0 0.5
  sleep 10
done
