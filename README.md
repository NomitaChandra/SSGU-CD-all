# SSGU-CD
<div style="display:none">Code for [](https://).</div>

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
Download address:
[PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
[SciBERT](https://huggingface.co/allenai/scibert_scivocab_cased)

## Dataset
The [CDR](https://pubmed.ncbi.nlm.nih.gov/26994911/) dataset and the [BioRED](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/) dataset can be downloaded.
Since [geniass](http://www.nactem.ac.uk/y-matsu/geniass/geniass-1.00.tar.gz) may not be available, we provide it in the repository.

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
 |    |-- cdr
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
```

## Training and Evaluation
Train the model on CDR and BioRED with the following command:

```bash
bash scripts/run_cdr.sh
bash scripts/run_biored_cd.sh
```

## Saving and Evaluating Models
You can save the model by setting the `--save_path` argument before training. After that, You can evaluate the saved model by setting the `--load_path` argument, then the code will skip training and evaluate the saved model on benchmarks.

## Test
You can train your own model or directly download our pre-trained model, which is available for download at [huggingface](https://huggingface.co/NNroc/SSGU-CD/tree/main).
Before running the shell script, you need to modify the `--load_path`, which represents the storage location of the model. You can also modify the `--save_result`, which represents the storage location for the output files.
```bash
bash scripts/test_cdr.sh
bash scripts/test_biored_cd.sh
```
