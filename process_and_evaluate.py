import argparse
import sys
import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from model_bio import Model
from utils import set_seed
from prepro_bio import read_bio
from save_result import Logger
from evaluation import train, evaluate, to_official_bio, gen_data_bio
import subprocess

original_dir = os.getcwd()

# Change to data_processing directory
os.chdir("/Users/kavithakamarthy/Downloads/SSGU-CD-all/SSGU-CD/data_processing")
subprocess.run(["python", "process_cdr.py", "/Users/kavithakamarthy/Downloads/SSGU-CD-all/SSGU-CD/dataset/cdr"])
os.chdir(original_dir)


import argparse
import sys
import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from model_bio import Model
from utils import set_seed
from prepro_bio import read_bio
from save_result import Logger
from evaluation import train, evaluate, to_official_bio, gen_data_bio

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="cdr", type=str)
parser.add_argument("--data_dir", default="./dataset/cdr", type=str)
parser.add_argument("--transformer_type", default="bert", type=str)
parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
parser.add_argument("--train_file", default="Train.BioC.JSON", type=str)
parser.add_argument("--dev_file", default="Dev.BioC.JSON", type=str)
parser.add_argument("--test_file", default="Test.BioC.JSON", type=str)
parser.add_argument("--save_path", default="out", type=str)
parser.add_argument("--load_path", default="", type=str)
parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--max_seq_length", default=1024, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                         "than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--train_batch_size", default=4, type=int, help="Batch size for training.")
parser.add_argument("--test_batch_size", default=8, type=int, help="Batch size for testing.")
parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Warm up ratio for Adam.")
parser.add_argument("--num_train_epochs", default=30.0, type=float, help="Total number of training epochs to perform.")
parser.add_argument("--evaluation_steps", default=-1, type=int, help="Number of training steps between evaluations.")
parser.add_argument("--seed", type=int, default=66, help="random seed for initialization")
parser.add_argument("--num_class", type=int, default=97, help="Number of relation types in dataset.")
parser.add_argument('--gnn', type=str, default='GCN', help="GCN/GAT")
parser.add_argument('--use_gcn', type=str, default='tree', help="use gcn, both/mentions/tree/false")
parser.add_argument('--dropout', type=float, default=0.5, help="0.0/0.2/0.5")
parser.add_argument('--loss', type=str, default='BSCELoss',
                    help="use BSCELoss/BalancedLoss/ATLoss/AsymmetricLoss/APLLoss")
parser.add_argument('--s0', type=float, default=0.3)
parser.add_argument("--demo", type=str, default='false', help='use a few data to test. default true/false')
parser.add_argument("--unet_in_dim", type=int, default=3, help="unet_in_dim.")
parser.add_argument("--unet_out_dim", type=int, default=256, help="unet_out_dim.")
parser.add_argument("--down_dim", type=int, default=256, help="down_dim.")
parser.add_argument("--bert_lr", default=3e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--max_height", type=int, default=64, help="max_height.")
parser.add_argument("--rel2", type=int, default=0, help="")
parser.add_argument("--save_result", type=str, default="", help="save predict result.")
args, _ = parser.parse_known_args()

if args.task == 'cdr':
    args.data_dir = './dataset/cdr'
    args.train_file = 'train_filter.data'
    args.dev_file = 'dev_filter.data'
    args.test_file = 'test.data'
    args.model_name_or_path = '/Users/kavithakamarthy/Downloads/SSGU-CD/dataset/cdr/data/pretrained/scibert_scivocab_cased'
    args.train_batch_size = 12
    args.test_batch_size = 12
    args.learning_rate = 2e-5
    args.num_class = 2
    args.num_train_epochs = 30

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
file_name = "{}_{}_{}_seed_{}_{}_{}_{}_{}".format(
    args.train_file.split('.')[0],
    args.transformer_type, args.data_dir.split('/')[-1],
    args.loss, args.use_gcn, args.s0, args.dropout, str(args.seed), )
args.save_path = os.path.join(args.save_path, file_name)
args.save_pubtator = os.path.join('./result/' + args.task + '/' + args.task  + '_' + args.loss
                                  + '_' + str(args.use_gcn) + '_s0=' + str(args.s0)
                                  + '_dropout=' + str(args.dropout) + '_' + str(args.seed))
if args.load_path == "":
    sys.stdout = Logger(stream=sys.stdout, filename=args.save_pubtator + '_test.log')
read = read_bio
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = torch.cuda.device_count()
args.device = device

config = AutoConfig.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path, num_labels=args.num_class, )
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, )
model = AutoModel.from_pretrained(
    args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config, )
config.cls_token_id = tokenizer.cls_token_id
config.sep_token_id = tokenizer.sep_token_id
config.transformer_type = args.transformer_type
set_seed(args)
model = Model(args, config, model, num_labels=1)
model.to('cpu')
model_path = '/Users/kavithakamarthy/Downloads/SSGU-CD-all/SSGU-CD/out/train_filter_bert_cdr_seed_BSCELoss_tree_0.3_0.5_66_best'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
test_file = os.path.join(args.data_dir, args.test_file)
test_features = read(args, test_file, tokenizer, max_seq_length=args.max_seq_length)
test_score, test_output, preds = evaluate(args, model, test_features, tag="test", generate=True)
print(test_output)