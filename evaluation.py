import os
import os.path
import json
import numpy as np
from prepro import relation2id_biored, id2relation_biored, biored_pairs, cdr_rel2id, gda_rel2id, cdr_id2rel, gda_id2rel
from zipfile import ZipFile

# rel2id = json.load(open('meta/rel2id.json', 'r'))
# id2rel = {value: key for key, value in rel2id.items()}
cdr_pairs = [("Chemical", "Disease"), ("Disease", "Chemical")]
gda_pairs = [("Gene", "Disease"), ("Disease", "Gene")]


def to_official(preds, features):
    h_idx, t_idx, title = [], [], []

    for f in features:
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if p != 0:
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2rel[p],
                    }
                )
    return res


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))
    return fact_in_train


def official_evaluate(tmp, path, tag, args):
    '''
        Adapted from the official evaluation code
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, args.train_file), truth_dir)
    fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    if tag == 'dev':
        truth = json.load(open(os.path.join(path, args.dev_file)))
    else:
        truth = json.load(open(os.path.join(path, args.test_file)))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}
    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set([1])
            tot_evidences += len([1])

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (
            len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (
            len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train, re_p, re_r


def to_official_bio(args, preds, features):
    h_idx, t_idx, title, h_idx_type, t_idx_type = [], [], [], [], []
    if args.task == 'cdr':
        bio_pairs = cdr_pairs
        id2rel = cdr_id2rel
    elif args.task == 'gda':
        bio_pairs = gda_pairs
        id2rel = gda_id2rel
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
                    print(h_idx_type[i], t_idx_type[i], 'no')
                    continue
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2rel[p].split(':')[1]
                    }
                )
    # res.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    return res


def gen_data_bio(args, res):
    save_path = './result/' + args.task
    if args.task == 'cdr':
        pubtator_test = './dataset/' + args.task + '/CDR_TestSet.PubTator.txt'
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
    # with ZipFile(save_path + '/output.zip', 'w') as z:
    #     z.write(save_path + '/output.pubtator')
