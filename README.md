# SSGU-CD
<div style="display:none">Code for [](https://arxiv.org/).</div>

Our code is modified based on [ATLOP](https://github.com/wzhouad/ATLOP), [DocuNet](https://github.com/zjunlp/DocuNet) and [UGDRE](https://github.com/QiSun123/UGDRE). Here we sincerely thanks for their excellent work.

## Environments
* Python (tested on 3.6.7)
* pytorch (tested in 1.7.1)
* CUDA (tested on 10.2)
```bash
conda create -n SSGU-CD python=3.6.7
conda activate SSGU-CD
pip install -r requirements.txt
```

## Pretraining
[PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
[SciBERT](https://huggingface.co/allenai/scibert_scivocab_cased)

## Dataset
The [CDR](https://pubmed.ncbi.nlm.nih.gov/26994911/) dataset and the [BioRED](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/) dataset can be downloaded following the instructions in [edge-oriented graph](https://github.com/fenchri/edge-oriented-graph).
Since [geniass]() may not be available, we provide it in the repository
You need to overwrite the files under ./data_processing to edge-oriented-graph/data_processing, use the following command to generate the data:
```shell
bash process_cdr.sh
bash process_biored_cd.sh
```

The expected structure of files is:
```
SSGU-CD
 |-- dataset
 |    |-- cdr
 |    |    |-- train_filter.data
 |    |    |-- dev_filter.data
 |    |    |-- test_filter.data
 |    |-- biored_cd
 |    |    |-- train+dev.data
 |    |    |-- test.data
```

## Training and Evaluation
Train the BERT model on CDR and BioRED with the following command:

```bash
sh scripts/run_cdr.sh
sh scripts/run_biored_cd.sh
```

## Saving and Evaluating Models
You can save the model by setting the `--save_path` argument before training. The model correponds to the best dev results will be saved. After that, You can evaluate the saved model by setting the `--load_path` argument, then the code will skip training and evaluate the saved model on benchmarks.