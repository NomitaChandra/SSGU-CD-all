#!/bin/sh

python train_bio.py \
          --task biored_cd \
          --loss "BSCELoss" \
          --use_gcn "tree" \
          --s0 0.3 \
          --dropout 0.5 \
          --rel2 1 \
          --load_path "./result/train+dev_biored_cd_both_rel2" \
          --save_result ""

python train_bio.py \
          --task biored_cd \
          --loss "BSCELoss" \
          --use_gcn "tree" \
          --s0 0.3 \
          --dropout 0.5 \
          --rel2 0 \
          --load_path "./result/train+dev_biored_cd_both" \
          --save_result ""
