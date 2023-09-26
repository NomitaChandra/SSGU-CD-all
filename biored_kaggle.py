import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from apex import amp
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
# from model import DocREModel
from model_biored import DocREModel
from utils import set_seed, collate_fn
from prepro import read_biored, read_biored_novel
from evaluation_biored_kaggle import write_in_file_kaggle, to_official_kaggle


def evaluate(args, model, features, tag="test"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds = []
    golds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            logits = model(**inputs)
            logits = logits.cpu().numpy()

            if args.isrank:
                pred = np.zeros((logits.shape[0], logits.shape[1]))
                for i in range(1, logits.shape[1]):
                    pred[(logits[:, i] > logits[:, 0]), i] = 1
                pred[:, 0] = (pred.sum(1) == 0)
            else:
                pred = np.zeros((logits.shape[0], logits.shape[1] + 1))
                for i in range(logits.shape[1]):
                    pred[(logits[:, i] > 0.), i + 1] = 1
                pred[:, 0] = (pred.sum(1) == 0)

            preds.append(pred)
            labels = [np.array(label, np.float32) for label in batch[2]]
            golds.append(np.concatenate(labels, axis=0))

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    # 从这里开始
    ans = to_official_kaggle(preds, features, args)
    write_in_file_kaggle(ans, args.data_dir, args)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/BioRED_Subtask1", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="Train.BioC.JSON", type=str)
    parser.add_argument("--dev_file", default="Dev.BioC.JSON", type=str)
    parser.add_argument("--test_file", default="bc8_biored_task1_val.json", type=str)
    parser.add_argument("--save_path", default="out", type=str)
    parser.add_argument("--load_path", default="./out/", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--isrank", type=int, default='1 means use ranking loss, 0 means not use')
    parser.add_argument("--m_tag", type=str, default='PN/PU/S-PU')
    parser.add_argument('--beta', type=float, default=0.0, help='beta of pu learning (default 0.0)')
    parser.add_argument('--gamma', type=float, default=1.0, help='gamma of pu learning (default 1.0)')
    parser.add_argument('--m', type=float, default=1.0, help='margin')
    parser.add_argument('--e', type=float, default=3.0, help='estimated a priors multiple')
    parser.add_argument("--novel", action="store_true", help='Whether to consider novel')

    parser.add_argument("--unet_in_dim", type=int, default=3,
                        help="unet_in_dim.")
    parser.add_argument("--unet_out_dim", type=int, default=256,
                        help="unet_out_dim.")
    parser.add_argument("--down_dim", type=int, default=256,
                        help="down_dim.")
    parser.add_argument("--channel_type", type=str, default='context-based',
                        help="unet_out_dim.")
    parser.add_argument("--bert_lr", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_height", type=int, default=50,
                        help="log.")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    file_name = "{}_{}_{}_{}_isrank_{}_m_{}_e_{}_seed_{}_{}".format(
        args.train_file.split('.')[0],
        args.transformer_type,
        args.data_dir.split('/')[-1],
        args.m_tag,
        str(args.isrank),
        args.m,
        args.e,
        str(args.seed),
        args.novel)
    print(args)

    if args.novel:
        read = read_biored_novel
        args.num_class = args.num_class * 2
    else:
        read = read_biored

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    test_file = os.path.join(args.data_dir, args.test_file)
    test_features, priors = read(args, test_file, tokenizer, max_seq_length=args.max_seq_length)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(args, config, priors, priors * args.e, model)
    model.to(0)

    # Testing
    args.load_path = os.path.join(args.load_path, file_name + '_best')
    print(args.load_path)
    model = amp.initialize(model, opt_level="O1", verbosity=0)
    model.load_state_dict(torch.load(args.load_path))
    evaluate(args, model, test_features, tag="test")
    # args.load_path = os.path.join(args.load_path, file_name + '_best')
    # print(args.load_path)
    # evaluate(args, model, test_features, tag="test")
    print('finish')


if __name__ == "__main__":
    main()
