#!/bin/sh

python train_bio.py \
          --task cdr \
          --loss "BSCELoss" \
          --use_gcn "tree" \
          --s0 0.3 \
          --dropout 0.5 \
          --load_path "/py_project/SSGU-CD/result/our_results/train_filter_cdr_tree" \
          --save_result ""
