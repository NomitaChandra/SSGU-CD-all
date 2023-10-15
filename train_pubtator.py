import argparse
import sys
import datetime
import os
import numpy as np
import torch
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
# from model import DocREModel
from model_bio import DocREModel
from utils import set_seed, collate_fn
from prepro import read_biored, read_cdr, read_gda
from save_result import Logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        score_best = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                      drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in tqdm(train_iterator, desc="epoch"):
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }

                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1

                if (step + 1) == len(train_dataloader) - 1 or (
                        args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    print("training risk:", loss.item(), "   step:", num_steps)

                    avg_val_risk = cal_val_risk(args, model, dev_features)
                    print('avg val risk:', avg_val_risk, '\n')

            # 进行每轮的评估测试
            torch.save(model.state_dict(), args.save_path)
            dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
            test_score, test_output = evaluate(args, model, test_features, tag="test")
            print("dev_score: ", dev_score, "dev_output: ", dev_output)
            print("test_score: ", test_score, "test_output: ", test_output)
            if score_best < dev_score:
                score_best = dev_score
                torch.save(model.state_dict(), args.save_path + '_best')

        return num_steps

    extract_layer = ["extractor", "bilinear"]
    bert_layer = ['model']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in bert_layer)], "lr": args.bert_lr},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in extract_layer)], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in extract_layer + bert_layer)]},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def cal_val_risk(args, model, features):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    val_risk = 0.
    nums = 0

    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'labels': batch[2],
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            risk, logits = model(**inputs)
            val_risk += risk.item()
            nums += 1

    return val_risk / nums


def evaluate(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds, golds = [], []
    for i, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'labels': batch[2],
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            output = model(**inputs)
            loss = output[0]
            pred = output[1].cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            golds.append(np.concatenate([np.array(label, np.float32) for label in batch[2]], axis=0))

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    golds = np.concatenate(golds, axis=0).astype(np.float32)
    tp = ((preds[:, 1] == 1) & (golds[:, 1] == 1)).astype(np.float32).sum()
    tn = ((golds[:, 1] == 1) & (preds[:, 1] != 1)).astype(np.float32).sum()
    fp = ((preds[:, 1] == 1) & (golds[:, 1] != 1)).astype(np.float32).sum()
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + tn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    output = {
        tag + "_F1": f1 * 100,
        tag + "_P": precision * 100,
        tag + "_R": recall * 100
    }
    return f1, output


# def evaluate(args, model, features, tag="test"):
#     dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
#                             drop_last=False)
#     preds = []
#     golds = []
#     for batch in dataloader:
#         model.eval()
#         inputs = {'input_ids': batch[0].to(args.device),
#                   'attention_mask': batch[1].to(args.device),
#                   'labels': batch[2],
#                   'entity_pos': batch[3],
#                   'hts': batch[4],
#                   }
#
#         with torch.no_grad():
#             logits = model(**inputs)
#             logits = logits.cpu().numpy()
#
#             if args.isrank:
#                 pred = np.zeros((logits.shape[0], logits.shape[1]))
#                 for i in range(1, logits.shape[1]):
#                     pred[(logits[:, i] > logits[:, 0]), i] = 1
#                 pred[:, 0] = (pred.sum(1) == 0)
#             else:
#                 pred = np.zeros((logits.shape[0], logits.shape[1] + 1))
#                 for i in range(logits.shape[1]):
#                     pred[(logits[:, i] > 0.), i + 1] = 1
#                 pred[:, 0] = (pred.sum(1) == 0)
#
#             preds.append(pred)
#             labels = [np.array(label, np.float32) for label in batch[2]]
#             golds.append(np.concatenate(labels, axis=0))
#
#     preds = np.concatenate(preds, axis=0).astype(np.float32)
#     # # 从这里开始
#     # ans = to_official(preds, features, args)
#     # re_f1, re_p, re_r, re_novel_f1, re_novel_p, re_novel_r = official_evaluate(ans, args.data_dir, tag, args)
#     # output = {
#     #     tag + "_F1": re_f1 * 100,
#     #     "re_p": re_p * 100,
#     #     "re_r": re_r * 100,
#     #     tag + "_novel_F1": re_novel_f1 * 100,
#     #     "re_novel_p": re_novel_p * 100,
#     #     "re_novel_r": re_novel_r * 100,
#     # }
#     # # print("my output:", output)
#     # return re_f1, output
#
#     # AI生成的答案
#     preds = preds[:, 1:]
#     # 样本中的标答
#     golds = np.concatenate(golds, axis=0).astype(np.float32)[:, 1:]
#
#     TPs = preds * golds  # (N, R)
#     TP = TPs.sum()
#     P = preds.sum()
#     T = golds.sum()
#
#     micro_p = TP / P if P != 0 else 0
#     micro_r = TP / T if T != 0 else 0
#     micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r > 0 else 0
#     mi_output = {
#         tag + "_F1": micro_f * 100,
#         "re_precision": micro_p * 100,
#         "re_recall": micro_r * 100,
#     }
#     return micro_f, mi_output


def main():
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
    parser.add_argument("--isrank", type=int, default='1', help='1 means use ranking loss, 0 means not use')
    parser.add_argument("--m_tag", type=str, default='S-PU', help='PN/PU/S-PU')
    parser.add_argument('--beta', type=float, default=0.0, help='beta of pu learning (default 0.0)')
    parser.add_argument('--gamma', type=float, default=1.0, help='gamma of pu learning (default 1.0)')
    parser.add_argument('--m', type=float, default=1.0, help='margin')
    parser.add_argument('--e', type=float, default=3.0, help='estimated a priors multiple')

    parser.add_argument("--unet_in_dim", type=int, default=3,
                        help="unet_in_dim.")
    parser.add_argument("--unet_out_dim", type=int, default=256,
                        help="unet_out_dim.")
    parser.add_argument("--down_dim", type=int, default=256,
                        help="down_dim.")
    parser.add_argument("--channel_type", type=str, default='context-based',
                        help="unet_out_dim.")
    parser.add_argument("--bert_lr", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_height", type=int, default=35,
                        help="log.")

    parser.add_argument("--tau", default=2.0, type=float, help="tau")
    parser.add_argument("--tau_base", default=1.0, type=float, help="tau_base")
    parser.add_argument("--lambda_1", default=1.0, type=float, help="lambda_1")
    parser.add_argument("--lambda_2", default=1.0, type=float, help="lambda_2")
    parser.add_argument("--lambda_3", default=1.0, type=float, help="lambda_3")
    parser.add_argument("--sample_ratio", default=1.0, type=float, help="sample_ratio")
    args = parser.parse_args()

    if args.task == 'cdr':
        args.data_dir = './dataset/cdr'
        args.train_file = 'train.data'
        args.dev_file = 'dev.data'
        args.test_file = 'test.data'
        # args.train_file = 'train_filter.data'
        # args.dev_file = 'dev_filter.data'
        # args.test_file = 'test_filter.data'
        args.model_name_or_path = '/home/yjs1217/Downloads/pretrained/scibert_scivocab_uncased'
        args.train_batch_size = 16
        args.test_batch_size = 16
        args.learning_rate = 2e-5
        args.num_class = 2
        args.num_train_epochs = 50
        read = read_cdr
    elif args.task == 'gda':
        args.data_dir = './dataset/gda'
        args.train_file = 'train.data'
        args.dev_file = 'dev.data'
        args.test_file = 'test.data'
        args.model_name_or_path = '/home/yjs1217/Downloads/pretrained/scibert_scivocab_uncased'
        args.train_batch_size = 4
        args.test_batch_size = 4
        args.learning_rate = 2e-5
        args.num_class = 2
        args.num_train_epochs = 50
        read = read_gda

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    file_name = "{}_{}_{}_seed_{}".format(
        args.train_file.split('.')[0],
        args.transformer_type,
        args.data_dir.split('/')[-1],
        str(args.seed))
    args.save_path = os.path.join(args.save_path, file_name)
    print(args)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    sys.stdout = Logger(stream=sys.stdout,
                        filename='./result/' + args.task + '/' + args.task + '_' + timestamp + '.log')

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

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)

    train_features, priors = read(args, train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features, _ = read(args, dev_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features, _ = read(args, test_file, tokenizer, max_seq_length=args.max_seq_length)

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

    print(args.m_tag, args.isrank)

    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features, test_features)

        print("BEST TEST")
        model.load_state_dict(torch.load(args.save_path + '_best'))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print(dev_output)
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print(test_output)

    else:  # Testing
        args.load_path = os.path.join(args.load_path, file_name)
        print(args.load_path)

        print("TEST")
        model.load_state_dict(torch.load(args.load_path))
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print(test_output)


if __name__ == "__main__":
    main()
