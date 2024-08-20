# SSGU-CD

Code for [](https://).

Our code is modified based on [ATLOP](https://github.com/wzhouad/ATLOP), [DocuNet](https://github.com/zjunlp/DocuNet)
and [UGDRE](https://github.com/QiSun123/UGDRE). Here we sincerely thanks for their excellent work.

## Environments

* Python (tested on 3.6.7)
* pytorch (tested in 1.7.1)
* CUDA (tested on 10.2)

Firstly, you need to download and enter the project folder SSGU-CD. Then, you need to create and enter the conda environment. The
command is as follows:

```bash
cd SSGU-CD
conda create -n SSGU-CD python=3.6.7
conda activate SSGU-CD
pip install -r requirements.txt
chmod +x ./data_preprocess/common/
```

## Dataset and Model

The [CDR](https://pubmed.ncbi.nlm.nih.gov/26994911/) dataset and
the [BioRED](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/) dataset can be downloaded.
Since [geniass](http://www.nactem.ac.uk/y-matsu/geniass/geniass-1.00.tar.gz) may not be available, we provide it in the
repository.
The [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract) pre-trained model can be
downloaded.
The [SciBERT](https://huggingface.co/allenai/scibert_scivocab_cased) pre-trained model can be downloaded.
You can download our pre-trained model, which is available for download
at [huggingface](https://huggingface.co/NNroc/SSGU-CD/tree/main).

You can regenerate the processed data using the following command, and then intermediate data (`{}.data`) can be
generated from `preprocess_cdr.sh` and `preprocess_biored_cd.sh`.

```shell
cd data_processing
bash process_cdr.sh
bash process_biored_cd.sh
cd ..
```

The expected structure of files is:

```
SSGU-CD
 |-- data_processing
 |    |-- process_biored_cd.sh
 |    |-- process_cdr.sh
 |-- dataset
 |    |    |-- CDR_TrainingSet.PubTator.txt
 |    |    |-- CDR_DevelopmentSet.PubTator.txt
 |    |    |-- CDR_TestSet.PubTator.txt
 |    |    |-- train_filter.data
 |    |    |-- dev_filter.data
 |    |    |-- test_filter.data
 |    |-- biored
 |    |    |-- Train.PubTator
 |    |    |-- Dev.PubTator
 |    |    |-- Test.PubTator
 |    |-- biored_cd
 |    |    |-- train+dev.data
 |    |    |-- test.data
 |-- result
 |    |    |-- train+dev_biored_cd_both
 |    |    |-- train+dev_biored_cd_both_rel2
 |    |    |-- train_filter_cdr_tree
```

## Training and Evaluation (in Jupyter)

Details of training and evaluation can be found in the `train_cdr.ipynb`, `train_biored.ipynb`, and `evaluate.ipynb` (not best results). You need to configure the conda environment first.
Among them, 10 documents in biored's training do not include chemical and disease entities. Therefore, the training set only consists of 490 documents.

## Training and Evaluation

Training models on CDR and BioRED with the following command:

```bash
python train_bio.py --task cdr
python train_bio.py --task biored_cd
```

Before running the shell script, you need to modify the `--load_path`, which represents the storage location of the
model. You can also modify the `--save_result`, which represents the storage location for the output files.

```bash
bash scripts/test_cdr.sh
bash scripts/test_biored_cd.sh
```
