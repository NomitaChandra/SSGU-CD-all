# SSGU-CD
<div style="display:none">Code for [](https://arxiv.org/).</div>

Our code is modified based on [ATLOP](https://github.com/wzhouad/ATLOP), [DocuNet](https://github.com/zjunlp/DocuNet) and [UGDRE](https://github.com/QiSun123/UGDRE). Here we sincerely thanks for their excellent work.

## Environments
* Python (tested on 3.6.7)
* CUDA (tested on 10.1)
```bash
conda create -n SSGU-CD python=3.6.7
conda activate SSGU-CD
pip install -r requirements.txt
```

## Dataset
The [CDR](https://pubmed.ncbi.nlm.nih.gov/26994911/) dataset can be downloaded following the instructions in [edge-oriented graph](https://github.com/fenchri/edge-oriented-graph). The expected structure of files is:
```
SSGU-CD
 |-- dataset
 |    |-- cdr
 |    |    |-- train_filter.data
 |    |    |-- dev_filter.data
 |    |    |-- test_filter.data
```

## Training and Evaluation
Train the BERT model on CDR with the following command:

```bash
sh scripts/run_cdr.sh
```

## Saving and Evaluating Models
You can save the model by setting the `--save_path` argument before training. The model correponds to the best dev results will be saved. After that, You can evaluate the saved model by setting the `--load_path` argument, then the code will skip training and evaluate the saved model on benchmarks.