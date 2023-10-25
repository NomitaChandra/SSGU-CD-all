# biored
Code for arxiv 2022 Main Conference paper [A ](https://arxiv.org/).

Our code is modified based on [Docunet](https://github.com/). Here we sincerely thanks for their excellent work.

## Requirements
* Python (tested on 3.6.7)
* CUDA (tested on 11.0)
* [PyTorch](http://pytorch.org/) (tested on 1.7.1)
* [Transformers](https://github.com/huggingface/transformers) (tested on 4.18.0)
* numpy (tested on 1.19.5)
* [apex](https://github.com/NVIDIA/apex) (tested on 0.1)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* ujson
* tqdm

## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data).

The [Re-DocRED](https://arxiv.org/abs/2205.12696) dataset can be downloaded following the instructions at [link](https://github.com/tonytan48/Re-DocRED).

The [ChemDisGene](https://arxiv.org/abs/2204.06584) dataset can be downloaded following the instructions at [link](https://github.com/chanzuckerberg/ChemDisGene).
```
biodre
 |-- dataset
 |    |-- chemdisgene
 |    |    |-- train.json
 |    |    |-- valid.json
 |    |    |-- test.anno_all.json
 |-- meta
 |    |-- rel2id.json
 |    |-- relation_map.json
```

## Training and Evaluation
### DocRED
Train DocRED model with the following command:

```bash
>> sh scripts/run_bert.sh  # S-PU BERT
```

### ChemDisGene
Train ChemDisGene model with the following command:
```bash
>> sh scripts/run_bio.sh  # S-PU PubmedBERT
>> sh scripts/run_bio_rank.sh  # SSR-PU PubmedBERT
```
