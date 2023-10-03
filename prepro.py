from tqdm import tqdm
import ujson as json
import numpy as np
import unidecode
from pubtator.convert_pubtator import load_pubtator_into_documents
import random
import copy
import os

docred_rel2id = json.load(open('meta/rel2id.json', 'r'))
ctd_rel2id = json.load(open('meta/relation_map.json', 'r'))
ENTITY_PAIR_TYPE_SET = set([("Chemical", "Disease"), ("Chemical", "Gene"), ("Gene", "Disease")])
relation2id_biored = {'1:NR:2': 0, '1:Association:2': 1, '1:Positive_Correlation:2': 2, '1:Bind:2': 3,
                      '1:Negative_Correlation:2': 4, '1:Comparison:2': 5, '1:Conversion:2': 6,
                      '1:Cotreatment:2': 7, '1:Drug_Interaction:2': 8}
id2relation_biored = {0: '1:NR:2', 1: '1:Association:2', 2: '1:Positive_Correlation:2', 3: '1:Bind:2',
                      4: '1:Negative_Correlation:2', 5: '1:Comparison:2', 6: '1:Conversion:2',
                      7: '1:Cotreatment:2', 8: '1:Drug_Interaction:2'}
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}

biored_pairs = [("DiseaseOrPhenotypicFeature", "GeneOrGeneProduct"),
                ("DiseaseOrPhenotypicFeature", "SequenceVariant"),
                ("ChemicalEntity", "ChemicalEntity"),
                ("ChemicalEntity", "GeneOrGeneProduct"),
                ("ChemicalEntity", "DiseaseOrPhenotypicFeature"),
                ("GeneOrGeneProduct", "GeneOrGeneProduct"),
                ("ChemicalEntity", "SequenceVariant"),
                ("SequenceVariant", "SequenceVariant")]
biored_entities = ["ChemicalEntity", "DiseaseOrPhenotypicFeature", "GeneOrGeneProduct", "SequenceVariant"]


def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res


def addEntitySentence(entities, curSent):
    vertexSentence = {}
    for enti in entities:
        for menti in enti:
            key = menti['sent_id'] * 10 + menti['pos'][0]
            vertexSentence[key] = copy.deepcopy(menti)
    vertexlists = sorted(vertexSentence.items(), key=lambda item: item[0])
    vertexsentencelist = []
    for item in vertexlists:
        vertexsentencelist += curSent[item[1]['sent_id']][item[1]['pos'][0]:item[1]['pos'][1]]
    return vertexsentencelist


def map_index(chars, tokens):
    # position index mapping from character level offset to token level offset
    ind_map = {}
    i, k = 0, 0  # (character i to token k)
    len_char = len(chars)
    num_token = len(tokens)
    while k < num_token:
        if i < len_char and chars[i].strip() == "":
            ind_map[i] = k
            i += 1
            continue
        token = tokens[k]
        if token[:2] == "##":
            token = token[2:]
        if token[:1] == "Ġ":
            token = token[1:]

        # assume that unk is always one character in the input text.
        if token != chars[i:(i + len(token))]:
            ind_map[i] = k
            i += 1
            k += 1
        else:
            for _ in range(len(token)):
                ind_map[i] = k
                i += 1
            k += 1

    return ind_map


# def read_docred(args, file_in, tokenizer, max_seq_length=1024):
#     i_line = 0
#     pos_samples = 0
#     neg_samples = 0
#     rel_nums = 0
#     features = []
#     if file_in == "":
#         return None
#     with open(file_in, "r") as fh:
#         data = json.load(fh)
#
#     re_fre = np.zeros(len(docred_rel2id) - 1)
#     for idx, sample in tqdm(enumerate(data), desc="Example"):
#         sents = []
#         sent_map = []
#
#         entities = sample['vertexSet']
#         entity_start, entity_end = [], []
#         for entity in entities:
#             for mention in entity:
#                 sent_id = mention["sent_id"]
#                 pos = mention["pos"]
#                 entity_start.append((sent_id, pos[0],))
#                 entity_end.append((sent_id, pos[1] - 1,))
#         for i_s, sent in enumerate(sample['sents']):
#             new_map = {}
#             for i_t, token in enumerate(sent):
#                 tokens_wordpiece = tokenizer.tokenize(token)
#
#                 if (i_s, i_t) in entity_start:
#                     tokens_wordpiece = ["*"] + tokens_wordpiece
#                 if (i_s, i_t) in entity_end:
#                     tokens_wordpiece = tokens_wordpiece + ["*"]
#                 new_map[i_t] = len(sents)
#                 sents.extend(tokens_wordpiece)
#             new_map[i_t + 1] = len(sents)
#             sent_map.append(new_map)
#
#         train_triple = {}
#         if "labels" in sample:
#             for label in sample['labels']:
#                 if 'evidence' not in label:
#                     evidence = []
#                 else:
#                     evidence = label['evidence']
#                 r = int(docred_rel2id[label['r']])
#                 re_fre[r - 1] += 1
#                 if (label['h'], label['t']) not in train_triple:
#                     train_triple[(label['h'], label['t'])] = [
#                         {'relation': r, 'evidence': evidence}]
#                 else:
#                     train_triple[(label['h'], label['t'])].append(
#                         {'relation': r, 'evidence': evidence})
#
#         entity_pos = []
#         for e in entities:
#             entity_pos.append([])
#             for m in e:
#                 start = sent_map[m["sent_id"]][m["pos"][0]]
#                 end = sent_map[m["sent_id"]][m["pos"][1]]
#                 entity_pos[-1].append((start, end,))
#
#         relations, hts = [], []
#         for h, t in train_triple.keys():
#             relation = [0] * len(docred_rel2id)
#             for mention in train_triple[h, t]:
#                 relation[mention["relation"]] = 1
#                 evidence = mention["evidence"]
#                 rel_nums += 1
#             relations.append(relation)
#             hts.append([h, t])
#             pos_samples += 1
#
#         for h in range(len(entities)):
#             for t in range(len(entities)):
#                 if h != t and [h, t] not in hts:
#                     relation = [1] + [0] * (len(docred_rel2id) - 1)
#                     relations.append(relation)
#                     hts.append([h, t])
#                     neg_samples += 1
#
#         assert len(relations) == len(entities) * (len(entities) - 1)
#
#         sents = sents[:max_seq_length - 2]
#         input_ids = tokenizer.convert_tokens_to_ids(sents)
#         input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
#
#         i_line += 1
#         feature = {'input_ids': input_ids,
#                    'entity_pos': entity_pos,
#                    'labels': relations,
#                    'hts': hts,
#                    'title': sample['title'],
#                    }
#         features.append(feature)
#
#     print("# of documents {}.".format(i_line))
#     print("# of positive examples {}.".format(pos_samples))
#     print("# of negative examples {}.".format(neg_samples))
#     re_fre = 1. * re_fre / (pos_samples + neg_samples)
#     print(re_fre)
#     print("# rels per doc", 1. * rel_nums / i_line)
#     return features, re_fre


def read_chemdisgene(args, file_in, tokenizer, max_seq_length=1024, lower=True):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    pos, neg, pos_labels, neg_labels = {}, {}, {}, {}
    for pair in list(ENTITY_PAIR_TYPE_SET):
        pos[pair] = 0
        neg[pair] = 0
        pos_labels[pair] = 0
        neg_labels[pair] = 0
    ent_nums = 0
    rel_nums = 0
    max_len = 0
    features = []
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    padid = tokenizer.pad_token_id
    cls_token_length = len(cls_token)
    print(cls_token, sep_token)
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    re_fre = np.zeros(len(ctd_rel2id))
    for idx, sample in tqdm(enumerate(data), desc="Example"):
        if "title" in sample and "abstract" in sample:
            text = sample["title"] + " " + sample["abstract"]
            if lower == True:
                text = text.lower()
        else:
            text = sample["text"]
            if lower == True:
                text = text.lower()

        text = unidecode.unidecode(text)
        tokens = tokenizer.tokenize(text)
        tokens = [cls_token] + tokens + [sep_token]
        text = cls_token + " " + text + " " + sep_token

        ind_map = map_index(text, tokens)

        entities = sample['entity']
        entity_start, entity_end = [], []

        train_triple = {}
        if "relation" in sample:
            for label in sample['relation']:
                if label['type'] not in ctd_rel2id:
                    continue
                if 'evidence' not in label:
                    evidence = []
                else:
                    evidence = label['evidence']
                r = int(ctd_rel2id[label['type']])

                if (label['subj'], label['obj']) not in train_triple:
                    train_triple[(label['subj'], label['obj'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['subj'], label['obj'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        entity_dict = {}
        entity2id = {}
        entity_type = {}
        eids = 0
        offset = 0

        for e in entities:

            entity_type[e["id"]] = e["type"]
            if e["start"] + cls_token_length in ind_map:
                startid = ind_map[e["start"] + cls_token_length] + offset
                tokens = tokens[:startid] + ['*'] + tokens[startid:]
                offset += 1
            else:
                continue
                startid = 0

            if e["end"] + cls_token_length in ind_map:
                endid = ind_map[e["end"] + cls_token_length] + offset
                endid += 1
                tokens = tokens[:endid] + ['*'] + tokens[endid:]
                endid += 1
                offset += 1
            else:
                continue
                endid = 0

            if startid >= endid:
                endid = startid + 1

            if e["id"] not in entity_dict:
                entity_dict[e["id"]] = [(startid, endid,)]
                entity2id[e["id"]] = eids
                eids += 1
                if e["id"] != "-":
                    ent_nums += 1
            else:
                entity_dict[e["id"]].append((startid, endid,))

        relations, hts = [], []
        for h, t in train_triple.keys():
            if h not in entity2id or t not in entity2id or (
                    (entity_type[h], entity_type[t]) not in ENTITY_PAIR_TYPE_SET):
                continue
            relation = [0] * (len(ctd_rel2id) + 1)
            for mention in train_triple[h, t]:
                if relation[mention["relation"] + 1] == 0:
                    re_fre[mention["relation"]] += 1
                relation[mention["relation"] + 1] = 1
                evidence = mention["evidence"]

            relations.append(relation)
            hts.append([entity2id[h], entity2id[t]])

            rel_num = sum(relation)
            rel_nums += rel_num
            pos_labels[(entity_type[h], entity_type[t])] += rel_num
            pos[(entity_type[h], entity_type[t])] += 1
            pos_samples += 1

        for h in entity_dict.keys():
            for t in entity_dict.keys():
                if (h != t) and ([entity2id[h], entity2id[t]] not in hts) and (
                        (entity_type[h], entity_type[t]) in ENTITY_PAIR_TYPE_SET) and (h != "-") and (t != "-"):
                    if (entity_type[h], entity_type[t]) not in neg:
                        neg[(entity_type[h], entity_type[t])] = 1
                    else:
                        neg[(entity_type[h], entity_type[t])] += 1

                    relation = [1] + [0] * (len(ctd_rel2id))
                    relations.append(relation)
                    hts.append([entity2id[h], entity2id[t]])
                    neg_samples += 1

        if len(tokens) > max_len:
            max_len = len(tokens)

        tokens = tokens[1:-1][:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1

        feature = {'input_ids': input_ids,
                   'entity_pos': list(entity_dict.values()),
                   'labels': relations,
                   'hts': hts,
                   'title': sample['docid'],
                   }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    re_fre = 1. * re_fre / (pos_samples + neg_samples)
    print(re_fre)
    print(max_len)
    print(pos)
    print(pos_labels)
    print(neg)
    print("# ents per doc", 1. * ent_nums / i_line)
    print("# rels per doc", 1. * rel_nums / i_line)
    return features, re_fre


def read_cdr(args, file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    re_fre = np.zeros(1)
    pos_samples = 0
    neg_samples = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = cdr_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(cdr_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                        if mention["relation"] != 0:
                            pos_samples += 1
                        else:
                            neg_samples += 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    re_fre[0] = pos_samples
    re_fre = 1. * re_fre / (pos_samples + neg_samples)
    return features, re_fre


def read_gda(args, file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    re_fre = np.zeros(1)
    pos_samples = 0
    neg_samples = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = gda_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(gda_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                        if mention["relation"] != 0:
                            pos_samples += 1
                        else:
                            neg_samples += 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    re_fre[0] = pos_samples
    re_fre = 1. * re_fre / (pos_samples + neg_samples)
    return features, re_fre


def read_biored(args, file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    re_fre = np.zeros(len(id2relation_biored))
    pos_samples = 0
    neg_samples = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]
            ent2ent_type = {}

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    if p[5] not in ent2ent_type:
                        ent2ent_type[p[5]] = p[7]
                    if p[11] not in ent2ent_type:
                        ent2ent_type[p[11]] = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents
                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = relation2id_biored[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(relation2id_biored)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                        if mention["relation"] != 0:
                            pos_samples += 1
                        else:
                            neg_samples += 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            assert len(ent2idx) == len(ent2ent_type)
            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           'ent2idx': ent2idx,
                           'ent2ent_type': ent2ent_type
                           }
                features.append(feature)
    re_fre[0] = pos_samples
    re_fre = 1. * re_fre / (pos_samples + neg_samples)
    # print(re_fre)
    # print("# rels per doc", 1. * pos_samples / len(features))
    # print("Number of documents: {}.".format(len(features)))
    # print("Max document length: {}.".format(maxlen))
    return features, re_fre


def get_pubtator(file_in):
    # 读取 pubtator 格式数据集
    all_documents = load_pubtator_into_documents(file_in, normalized_type_dict={}, re_id_spliter_str=r'\|')
    data = all_documents
    return data


#
# def read_pubtator(args, file_in, tokenizer, max_seq_length=1024, lower=True):
#     i_line = 0
#     pos_samples = 0
#     neg_samples = 0
#     ent_nums = 0
#     rel_nums = 0
#     max_len = 0
#     features = []
#     cls_token = tokenizer.cls_token
#     sep_token = tokenizer.sep_token
#     padid = tokenizer.pad_token_id
#     cls_token_length = len(cls_token)
#     print(cls_token, sep_token)
#     relation2id = relation2id_cdr
#     id2relation = id2relation_cdr
#
#     # 读取 pubtator 格式数据集
#     all_documents = load_pubtator_into_documents(file_in, normalized_type_dict={}, re_id_spliter_str=r'\|')
#     data = all_documents
#
#     re_fre = np.zeros(len(relation2id))
#     for idx, sample in tqdm(enumerate(data), desc="Example"):
#         if "passages" in sample:
#             text = sample["passages"][0]["text"] + " " + sample["passages"][1]["text"]
#             if lower == True:
#                 text = text.lower()
#         else:
#             text = sample["passages"][0]["text"]
#             if lower == True:
#                 text = text.lower()
#
#         text = unidecode.unidecode(text)
#         tokens = tokenizer.tokenize(text)
#         tokens = [cls_token] + tokens + [sep_token]
#         text = cls_token + " " + text + " " + sep_token
#         ind_map = map_index(text, tokens)
#         entities = sample['passages'][0]['annotations'] + sample['passages'][1]['annotations']
#
#         train_triple = {}
#         if "relations" in sample:
#             for label in sample['relations']:
#                 if label['infons']['type'] not in relation2id:
#                     print(label['infons']['type'], 'not in relation2id')
#                     continue
#                 if 'novel' not in label:
#                     novel = 'No'
#                 else:
#                     novel = label['infons']['novel']
#                 r = int(relation2id[label['infons']['type']])
#                 re_fre[r] += 1
#                 if (label['infons']['entity1'], label['infons']['entity2']) not in train_triple:
#                     train_triple[(label['infons']['entity1'], label['infons']['entity2'])] = [
#                         {'relation': r, 'novel': novel}]
#                 else:
#                     train_triple[(label['infons']['entity1'], label['infons']['entity2'])].append(
#                         {'relation': r, 'novel': novel})
#
#         entity_dict = {}
#         entity2id = {}
#         entity_type = {}
#         eids = 0
#         offset = 0
#
#         for e in entities:
#             e_start = int(e['locations'][0]['offset'])
#             e_end = int(e['locations'][0]['offset']) + int(e['locations'][0]['length'])
#             e_id = e['infons']['identifier']
#             e_ids = e_id.split(',')
#             entity_type[e_id] = e['infons']['type']
#             if e_start + cls_token_length in ind_map:
#                 startid = ind_map[e_start + cls_token_length] + offset
#                 tokens = tokens[:startid] + ['*'] + tokens[startid:]
#                 offset += 1
#             else:
#                 continue
#                 startid = 0
#
#             if e_end + cls_token_length in ind_map:
#                 endid = ind_map[e_end + cls_token_length] + offset
#                 endid += 1
#                 tokens = tokens[:endid] + ['*'] + tokens[endid:]
#                 endid += 1
#                 offset += 1
#             else:
#                 continue
#                 endid = 0
#
#             if startid >= endid:
#                 endid = startid + 1
#             for e_id in e_ids:
#                 if e_id not in entity_dict:
#                     entity_dict[e_id] = [(startid, endid,)]
#                     entity2id[e_id] = eids
#                     eids += 1
#                     if e_id != "-":
#                         ent_nums += 1
#                 else:
#                     entity_dict[e_id].append((startid, endid,))
#
#         relations, hts = [], []
#         for h, t in train_triple.keys():
#             if h not in entity2id or t not in entity2id:
#                 print('error: ', h, t)
#                 continue
#             relation = [0] * (len(relation2id) + 1)
#             for mention in train_triple[h, t]:
#                 relation[mention["relation"] + 1] = 1
#
#             relations.append(relation)
#             hts.append([entity2id[h], entity2id[t]])
#             pos_samples += 1
#
#             rel_num = sum(relation)
#             rel_nums += rel_num
#
#         for h in entity_dict.keys():
#             for t in entity_dict.keys():
#                 if [entity2id[h], entity2id[t]] not in hts:
#                     relation = [1] + [0] * (len(relation2id))
#                     relations.append(relation)
#                     hts.append([entity2id[h], entity2id[t]])
#                     neg_samples += 1
#
#         if len(tokens) > max_len:
#             max_len = len(tokens)
#
#         tokens = tokens[1:-1][:max_seq_length - 2]
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#         input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
#
#         i_line += 1
#
#         feature = {'input_ids': input_ids,
#                    'entity_pos': list(entity_dict.values()),
#                    'labels': relations,
#                    'hts': hts,
#                    'title': sample['id'],
#                    'id2entity': dict(zip(entity2id.values(), entity2id.keys()))
#                    }
#         features.append(feature)
#
#     print("# of documents {}.".format(i_line))
#     print("# of positive examples {}.".format(pos_samples))
#     print("# of negative examples {}.".format(neg_samples))
#     re_fre = 1. * re_fre / (pos_samples + neg_samples)
#     print(re_fre)
#     print("# rels per doc", 1. * rel_nums / i_line)
#     return features, re_fre


def read_docred(meta, file_in, tokenizer, max_seq_length=1024):
    docred_rel2id = json.load(open(meta, 'r'))
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)
    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []
        # words = []
        # token_map = []
        # lengthofPice = 0
        # mentions = []
        # inter_mask = []
        # for i in range(0, len(sample['vertexSet'])):
        #     inter_mask.append([1] * len(sample['vertexSet']))
        # for i in range(0, len(sample['vertexSet'])):
        #     inter_mask[i][i] = 0
        # for i, ent1 in enumerate(sample['vertexSet']):
        #     for j, ent2 in enumerate(sample['vertexSet']):
        #         if i != j:
        #             breakFlag = 0
        #             for men1 in ent1:
        #                 for men2 in ent2:
        #                     if men1['sent_id'] == men2['sent_id']:
        #                         inter_mask[i][j] = 0
        #                         inter_mask[j][i] = 0
        #                         breakFlag = 1
        #                         break
        #                 if breakFlag == 1:
        #                     break
        entities = sample['vertexSet']
        # vertexsentencelist = addEntitySentence(entities, sample['sents'])
        entity_start, entity_end = {}, {}
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                # mentions.append(pos)

                entity_start[(sent_id, pos[0],)] = "*"
                entity_end[(sent_id, pos[1] - 1,)] = "*"

        newsents = sample['sents']
        # newsents.append(vertexsentencelist)

        for i_s, sent in enumerate(newsents):
            new_map = {}
            for i_t, token in enumerate(sent):

                # oneToken = []
                # words.append(token)
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start.keys():
                    # tokens_wordpiece = [entity_start[(i_s, i_t)]] + tokens_wordpiece
                    # oneToken.append(lengthofPice + 1)
                    # lengthofPice += len(tokens_wordpiece)
                    # oneToken.append(lengthofPice)

                elif (i_s, i_t) in entity_end:
                    # tokens_wordpiece = tokens_wordpiece + [entity_end[(i_s, i_t)]]
                    # oneToken.append(lengthofPice)
                    # lengthofPice += len(tokens_wordpiece)
                    # oneToken.append(lengthofPice - 1)
                else:
                    # oneToken.append(lengthofPice)
                    # lengthofPice += len(tokens_wordpiece)
                    # oneToken.append(lengthofPice)
                # new_map[i_t] = len(sents)
                # sents.extend(tokens_wordpiece)
                # token_map.append(oneToken)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        # sent_occur = {}
        ei = 0
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))
                # if m["sent_id"] not in sent_occur.keys():
                #     sent_occur[m["sent_id"]] = []
                # if ei not in sent_occur[m["sent_id"]]:
                #     sent_occur[m["sent_id"]].append(ei)
            ei += 1
        relations, hts = [], []
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)
        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        # A = []
        # for i in range(0, len(input_ids)):
        #     A.append([0] * len(input_ids))
        # offset = 1
        # edges = 0
        # for token_s in token_map:
        #     start = token_s[0] + offset
        #     end = token_s[1] + offset
        #     for i in range(start, end):
        #         for j in range(start, end):
        #             if i < (len(input_ids) - 1) and j < (len(input_ids) - 1):
        #                 if A[i][j] == 0:
        #                     A[i][j] = 1
        #                     edges += 1
        # mentionsofPice = []
        # for ment in mentions:
        #     mentionsofPice.append([token_map[ment[0]][0], token_map[ment[1] - 1][1]])
        # for ment in mentionsofPice:
        #     start = ment[0] + offset
        #     end = ment[1] + offset
        #     for i in range(start, end):
        #         for j in range(start, end):
        #             if i < (len(input_ids) - 1) and j < (len(input_ids) - 1):
        #                 if A[i][j] == 0:
        #                     A[i][j] = 1
        #                     edges += 1
        # entityofPice = []
        # for ent in entity_pos:
        #     oneEntityP = []
        #     for ment in ent:
        #         if (ment[0] + offset) == (ment[1] - offset):
        #             oneEntityP.append(ment[0] + offset)
        #         for i in range(ment[0] + offset, ment[1] - offset):
        #             oneEntityP.append(i)
        #     entityofPice.append(oneEntityP)
        # predicted_Doc2 = []
        # for h in range(0, len(entities)):
        #     item = [0, h]
        #     predicted_Doc2.append(item)
        #
        # predictedEntityPairPiece = []
        # for item in predicted_Doc2:
        #     one_predicted = entityofPice[item[0]] + entityofPice[item[1]]
        #     predictedEntityPairPiece.append(one_predicted)
        #
        # for line in predictedEntityPairPiece:
        #     for i in line:
        #         for j in line:
        #             if A[i + offset][j + offset] == 0:
        #                 A[i + offset][j + offset] = 1
        #                 edges += 1
        #
        # for i in range(0, len(A)):
        #     A[i][i] = 1

        i_line += 1
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'title': sample['title'],
                   'Adj': A,
                   'inter_mask': inter_mask,
                   }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features
