import os.path
import numpy as np
import torch
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import set_seed, collate_fn
from prepro_bio import cdr_id2rel, biored_cd_id2rel, biored_id2rel

cdr_pairs = [("Chemical", "Disease"), ("Disease", "Chemical")]
biored_cd_pairs = [("ChemicalEntity", "DiseaseOrPhenotypicFeature"), ("DiseaseOrPhenotypicFeature", "ChemicalEntity")]
gda_pairs = [("Gene", "Disease"), ("Disease", "Gene")]
biored_pairs = [("ChemicalEntity", "ChemicalEntity"),
                ("GeneOrGeneProduct", "GeneOrGeneProduct"),
                ("ChemicalEntity", "DiseaseOrPhenotypicFeature"), ("DiseaseOrPhenotypicFeature", "ChemicalEntity"),
                ("ChemicalEntity", "GeneOrGeneProduct"), ("GeneOrGeneProduct", "ChemicalEntity"),
                ("ChemicalEntity", "SequenceVariant"), ("SequenceVariant", "ChemicalEntity"),
                ("GeneOrGeneProduct", "DiseaseOrPhenotypicFeature"),
                ("DiseaseOrPhenotypicFeature", "GeneOrGeneProduct"),
                ("DiseaseOrPhenotypicFeature", "SequenceVariant"), ("SequenceVariant", "DiseaseOrPhenotypicFeature")]


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
                list_feature_id = torch.tensor([i for i in range(args.train_batch_size)])
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          'adj_mention': batch[5].to(args.device),
                          'adj_syntactic_dependency_tree': batch[6].to(args.device),
                          'list_feature_id': list_feature_id.to(args.device)
                          }
                outputs = model(**inputs)
                outputs[0] = torch.sum(outputs[0], dim=0)
                loss = outputs[0] / args.gradient_accumulation_steps
                loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0
                                                               and num_steps % args.evaluation_steps == 0
                                                               and step % args.gradient_accumulation_steps == 0):
                    print("training risk:", loss.item(), "   step:", num_steps)
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


def evaluate(args, model, features, tag="dev", generate=False):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds, golds = [], []
    for i, batch in enumerate(dataloader):
        model.eval()
        list_feature_id = torch.tensor([i for i in range(args.train_batch_size)])
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'labels': batch[2],
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'adj_mention': batch[5].to(args.device),
                  'adj_syntactic_dependency_tree': batch[6].to(args.device),
                  'list_feature_id': list_feature_id.to(args.device)
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
    re_correct = 0
    preds_ans = to_official_bio(args, preds, features)
    golds_ans = to_official_bio(args, golds, features)
    if generate:
        gen_data_bio(args, preds_ans)
    for pred in preds_ans:
        if pred in golds_ans:
            re_correct += 1
    re_correct_2 = 0
    preds_ans_2 = []
    golds_ans_2 = []
    for pred in preds_ans:
        if [pred['title'], pred['h_idx'], pred['t_idx'], pred['r']] not in preds_ans_2 \
                and [pred['title'], pred['t_idx'], pred['h_idx'], pred['r']] not in preds_ans_2:
            preds_ans_2.append([pred['title'], pred['h_idx'], pred['t_idx'], pred['r']])
    for gold in golds_ans:
        if [gold['title'], gold['h_idx'], gold['t_idx'], gold['r']] not in golds_ans_2 \
                and [gold['title'], gold['t_idx'], gold['h_idx'], gold['r']] not in golds_ans_2:
            golds_ans_2.append([gold['title'], gold['h_idx'], gold['t_idx'], gold['r']])
    if args.task == 'biored':
        for pred in preds_ans_2:
            if pred in golds_ans_2 or [pred[0], pred[2], pred[1], pred[3]] in golds_ans_2:
                re_correct_2 += 1
        precision_2 = re_correct_2 / (len(preds_ans) + 1e-5)
        recall_2 = re_correct_2 / (len(golds_ans) + 1e-5)
        f1_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2 + 1e-5)
        output = {
            tag + "_F1": f1_2 * 100,
            tag + "_P": precision_2 * 100,
            tag + "_R": recall_2 * 100
        }
        return f1_2, output
    precision = re_correct / (len(preds_ans) + 1e-5)
    recall = re_correct / (len(golds_ans) + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    output = {
        tag + "_F1": f1 * 100,
        tag + "_P": precision * 100,
        tag + "_R": recall * 100
    }
    return f1, output


def to_official_bio(args, preds, features):
    h_idx, t_idx, title, h_idx_type, t_idx_type = [], [], [], [], []
    if args.task == 'cdr':
        bio_pairs = cdr_pairs
        id2rel = cdr_id2rel
    elif args.task == 'biored_cd':
        bio_pairs = biored_cd_pairs
        id2rel = biored_cd_id2rel
    elif args.task == 'biored':
        bio_pairs = biored_pairs
        id2rel = biored_id2rel
    else:
        return

    for f in features:
        hts = f["hts"]
        ent2ent_type = f["ent2ent_type"]
        ent2idx = f["ent2idx"]
        idx2ent = {v: k for k, v in ent2idx.items()}

        h_idx += [idx2ent[ht[0]] for ht in hts]
        h_idx_type += [ent2ent_type[idx2ent[ht[0]]] for ht in hts]
        t_idx += [idx2ent[ht[1]] for ht in hts]
        t_idx_type += [ent2ent_type[idx2ent[ht[1]]] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if p != 0:
                if (h_idx_type[i], t_idx_type[i]) not in bio_pairs and (
                        t_idx_type[i], h_idx_type[i]) not in bio_pairs:
                    # print(h_idx_type[i], t_idx_type[i], 'no')
                    continue
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2rel[p].split(':')[1]
                    }
                )
    return res


def gen_data_bio(args, res):
    save_path = './result/' + args.task
    print('generate predict result in ' + args.save_pubtator)
    if args.task == 'cdr':
        pubtator_test = './dataset/' + args.task + '/CDR_TestSet.PubTator.txt'
    elif args.task == 'biored_cd':
        pubtator_test = './dataset/' + args.task + '/Test.pubtator'
    else:
        return
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    origin = []
    pmid = ''
    with open(pubtator_test, 'r') as f:
        use = []
        for line in f.readlines():
            if pmid == '':
                pmid = line.split('|')[0]
            if line != '\n':
                use.append(line)
            else:
                origin.append({pmid: use})
                use = []
                pmid = ''
    with open(args.save_pubtator + '.pubtator', 'w') as f:
        for i in origin:
            pmid = next(iter(i))
            for line in i[pmid]:
                f.write(line)
            for x in res:
                if x['title'] == pmid:
                    f.write(x['title'] + '\t' + x['r'] + '\t' + x['h_idx'] + '\t' + x['t_idx'] + '\tpredict' + '\n')
            f.write('\n')
    print(args.save_pubtator + '.pubtator')
