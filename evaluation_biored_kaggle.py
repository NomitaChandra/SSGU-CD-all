import os
import os.path
import json
import numpy as np
from zipfile import ZipFile

relation2id_biored = {'Association': 0, 'Positive_Correlation': 1, 'Bind': 2, 'Negative_Correlation': 3,
                      'Comparison': 4,
                      'Conversion': 5, 'Cotreatment': 6, 'Drug_Interaction': 7}
id2relation_biored = {0: 'Association', 1: 'Positive_Correlation', 2: 'Bind', 3: 'Negative_Correlation',
                      4: 'Comparison',
                      5: 'Conversion', 6: 'Cotreatment', 7: 'Drug_Interaction'}

# [0]对应no，[1]对应novel
relation2id_biored_novel = {'Association': [0, 1],
                            'Positive_Correlation': [2, 3],
                            'Bind': [4, 5],
                            'Negative_Correlation': [6, 7],
                            'Comparison': [8, 9],
                            'Conversion': [10, 11],
                            'Cotreatment': [12, 13],
                            'Drug_Interaction': [14, 15]}
id2relation_biored_novel = {0: ['Association', 0], 1: ['Association', 1],
                            2: ['Positive_Correlation', 0], 3: ['Positive_Correlation', 1],
                            4: ['Bind', 0], 5: ['Bind', 1],
                            6: ['Negative_Correlation', 0], 7: ['Negative_Correlation', 1],
                            8: ['Comparison', 0], 9: ['Comparison', 1],
                            10: ['Conversion', 0], 11: ['Conversion', 1],
                            12: ['Cotreatment', 0], 13: ['Cotreatment', 1],
                            14: ['Drug_Interaction', 0], 15: ['Drug_Interaction', 1]}

biored_pairs = [("DiseaseOrPhenotypicFeature", "GeneOrGeneProduct"),
                ("DiseaseOrPhenotypicFeature", "SequenceVariant"),
                ("ChemicalEntity", "ChemicalEntity"),
                ("ChemicalEntity", "GeneOrGeneProduct"),
                ("ChemicalEntity", "DiseaseOrPhenotypicFeature"),
                ("GeneOrGeneProduct", "GeneOrGeneProduct"),
                ("ChemicalEntity", "SequenceVariant")]


def to_official_kaggle(preds, features, args):
    h_idx, t_idx, title, h_idx_type, t_idx_type = [], [], [], [], []

    for f in features:
        hts = f["hts"]
        id2entity = f["id2entity"]
        h_idx += [id2entity[ht[0]] for ht in hts]
        h_idx_type += [f["id2entity_type"][id2entity[ht[0]]] for ht in hts]
        t_idx += [id2entity[ht[1]] for ht in hts]
        t_idx_type += [f["id2entity_type"][id2entity[ht[1]]] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if p != 0:
                novel = ''
                r = ''
                if args.novel:
                    if id2relation_biored_novel[p - 1][1] == 0:
                        novel = 'No'
                    elif id2relation_biored_novel[p - 1][1] == 1:
                        novel = 'Novel'
                    r = id2relation_biored_novel[p - 1][0]
                else:
                    r = id2relation_biored[p - 1]
                if (h_idx_type[i], t_idx_type[i]) not in biored_pairs:
                    print(h_idx_type[i], t_idx_type[i], r)
                    continue
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': r,
                        'novel': novel
                    }
                )
    return res


def write_in_file_kaggle(tmp, path, args):
    '''
        Adapted from the official evaluation code
    '''
    truth = json.load(open(os.path.join(path, args.test_file)))
    # std 存储标答的关系四元组
    std = {}
    titleset = set([])
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r'], x['novel']))
    # ai 生成的
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r'], x['novel']) != (
                y['title'], y['h_idx'], y['t_idx'], y['r'], y['novel']):
            submission_answer.append(tmp[i])

    documents = truth['documents']
    for x in documents:
        title = x['id']
        relations = []
        for i in submission_answer:
            if i['title'] == title:
                novel = ''
                if i['novel'] != '':
                    novel = i['novel']
                else:
                    novel = 'Novel'
                relations.append(
                    {
                        "infons": {
                            "entity1": i['h_idx'],
                            "entity2": i['t_idx'],
                            "type": i['r'],
                            "novel": novel
                        }
                    }
                )
        x['relations'] = relations

    save_path = './result/bc8_biored_task'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + '/bc8_biored_task1_val_out.json', 'w') as f:
        json.dump(truth, f, indent=4, ensure_ascii=False)
    with ZipFile(save_path + '/output.zip', 'w') as z:
        z.write(save_path + '/bc8_biored_task1_val_out.json')
