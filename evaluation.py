import os
import os.path
import json
import numpy as np
from prepro_bio import cdr_rel2id, cdr_id2rel, biored_cd_rel2id, biored_cd_id2rel
from zipfile import ZipFile

# rel2id = json.load(open('meta/rel2id.json', 'r'))
# id2rel = {value: key for key, value in rel2id.items()}
cdr_pairs = [("Chemical", "Disease"), ("Disease", "Chemical")]
biored_cd_pairs = [("ChemicalEntity", "DiseaseOrPhenotypicFeature"), ("DiseaseOrPhenotypicFeature", "ChemicalEntity")]
gda_pairs = [("Gene", "Disease"), ("Disease", "Gene")]


def to_official_bio(args, preds, features):
    h_idx, t_idx, title, h_idx_type, t_idx_type = [], [], [], [], []
    if args.task == 'cdr':
        bio_pairs = cdr_pairs
        id2rel = cdr_id2rel
    elif args.task == 'biored_cd':
        bio_pairs = biored_cd_pairs
        id2rel = biored_cd_id2rel
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