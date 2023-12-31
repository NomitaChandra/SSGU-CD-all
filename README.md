# DGUNet-CD
<div style="display:none">Code for arxiv 2022 Main Conference paper [A ](https://arxiv.org/).</div>

Our code is modified based on [ATLOP](https://github.com/wzhouad/ATLOP), [DocuNet](https://github.com/zjunlp/DocuNet) and [UGDRE](https://github.com/QiSun123/UGDRE). Here we sincerely thanks for their excellent work.

## Requirements
* Python (tested on 3.6.7)
* CUDA (tested on 11.0)
* [PyTorch](http://pytorch.org/) (tested on 1.7.1)
* [Transformers](https://github.com/huggingface/transformers) (tested on 4.18.0)
* numpy (tested on 1.19.5)
* spacy (tested on 3.0.9)
* [apex](https://github.com/NVIDIA/apex) (tested on 0.1)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* ujson
* tqdm

## Dataset
The [CDR](https://arxiv.org/abs/2204.06584) dataset can be downloaded following the instructions at [link](https://github.com/chanzuckerberg/ChemDisGene).
```
biodre
 |-- dataset
 |    |-- cdr
 |    |    |-- train_filter.data
 |    |    |-- dev_filter.data
 |    |    |-- test_filter.data
```

## Training and Evaluation

### CDR
Train ChemDisGene model with the following command:
```bash
>> sh scripts/run_bio.sh
```