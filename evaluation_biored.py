import os
import os.path
import json
import numpy as np
from prepro import relation2id_biored, id2relation_biored, biored_pairs
from zipfile import ZipFile


def to_official(preds, features):
    h_idx, t_idx, title, h_idx_type, t_idx_type = [], [], [], [], []

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
                if (h_idx_type[i], t_idx_type[i]) not in biored_pairs and (
                        t_idx_type[i], h_idx_type[i]) not in biored_pairs:
                    print(h_idx_type[i], t_idx_type[i], 'no')
                    continue
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2relation_biored[p].split(':')[1]
                    }
                )
    # res.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    return res


def gen_data(res):
    save_path = './result/bc8_biored_task'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    origin = []
    pmid = ''
    with open('./dataset/BioRED_Subtask1/bc8_biored_task1_val.pubtator', 'r') as f:
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
    with open(save_path + '/bc8_biored_task1.pubtator', 'w') as f:
        for i in origin:
            pmid = next(iter(i))
            for line in i[pmid]:
                f.write(line)
            for x in res:
                if x['title'] == pmid:
                    f.write(x['title'] + '\t' + x['r'] + '\t' + x['h_idx'] + '\t' + x['t_idx'] + '\tNovel' + '\n')
            f.write('\n')
    with ZipFile(save_path + '/output.zip', 'w') as z:
        z.write(save_path + '/bc8_biored_task1.pubtator')
