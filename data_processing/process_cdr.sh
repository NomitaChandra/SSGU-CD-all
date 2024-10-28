#!/bin/bash

# Check if the input path argument is provided
if [ -z "$1" ]; then
    echo "Usage: bash file.sh <path_to_input_folder>"
    exit 1
fi

# Set the input path from the first argument
input_path=$1

for d in "Test";
do
    python3 process.py --input_file ${input_path}/CDR_${d}Set.PubTator.txt \
                       --output_file ${input_path}/${d} \
                       --data CDR

    # python3 filter_hypernyms.py --mesh_file 2017MeshTree.txt \
    #                             --input_file ${input_path}/${d}.data \
    #                             --output_file ${input_path}/${d}_filter.data
done

# mv ${input_path}/Training.data ${input_path}/train.data
# mv ${input_path}/Development.data ${input_path}/dev.data
mv ${input_path}/Test.data ${input_path}/test.data

# mv ${input_path}/Training_filter.data ${input_path}/train_filter.data
# mv ${input_path}/Development_filter.data ${input_path}/dev_filter.data
# mv ${input_path}/Test_filter.data ${input_path}/test_filter.data

# # Merge train and dev
# cat ${input_path}/train_filter.data > ${input_path}/train+dev_filter.data
# cat ${input_path}/dev_filter.data >> ${input_path}/train+dev_filter.data

