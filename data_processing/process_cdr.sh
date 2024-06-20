#!/usr/bin/env bash

for d in "Training" "Development" "Test";
do
    python3 process.py --input_file ../dataset/cdr/CDR_${d}Set.PubTator.txt \
                       --output_file ../dataset/cdr/${d} \
                       --data CDR

    python3 filter_hypernyms.py --mesh_file 2017MeshTree.txt \
                                --input_file ../dataset/cdr/${d}.data \
                                --output_file ../dataset/cdr/${d}_filter.data
done

mv ../dataset/cdr/Training.data ../dataset/cdr/train.data
mv ../dataset/cdr/Development.data ../dataset/cdr/dev.data
mv ../dataset/cdr/Test.data ../dataset/cdr/test.data

mv ../dataset/cdr/Training_filter.data ../dataset/cdr/train_filter.data
mv ../dataset/cdr/Development_filter.data ../dataset/cdr/dev_filter.data
mv ../dataset/cdr/Test_filter.data ../dataset/cdr/test_filter.data

# merge train and dev
cat ../dataset/cdr/train_filter.data > ../dataset/cdr/train+dev_filter.data
cat ../dataset/cdr/dev_filter.data >> ../dataset/cdr/train+dev_filter.data
