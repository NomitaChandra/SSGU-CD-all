#!/bin/sh

losses=(BalancedLoss ATLoss AsymmetricLoss APLLoss BSCELoss)
#use_gcns=("both" "mentions" "tree" "false")
losses="APLLoss"
use_gcns="tree"
for loss in "${losses[@]}"
do
  for use_gcn in "${use_gcns[@]}"
  do
    for ((i=1;i<=5;i++))
    do
    python train_bio.py --task cdr --loss $loss --use_gcn $use_gcn
    sleep 10
    done
  done
done