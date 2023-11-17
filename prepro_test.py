from tqdm import tqdm
import ujson as json
import numpy as np
import unidecode
from pubtator.convert_pubtator import load_pubtator_into_documents
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
import random
import copy
import re
import os
import spacy
import pickle

ENTITY_PAIR_TYPE_SET = set([("Chemical", "Disease"), ("Chemical", "Gene"), ("Gene", "Disease")])
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}


def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        # Avoid zero-length tokens
        for i, word in enumerate(words):
            if word == "":
                words[i] = " "
                spaces[i] = False
        # Remove the final trailing space
        if words[-1] == " ":
            words = words[0:-1]
            spaces = spaces[0:-1]
        else:
            spaces[-1] = False

        return Doc(self.vocab, words=words, spaces=spaces)


def addEntitySentence(entities, curSent):
    sents = []
    for i in curSent:
        sents += i
    vertexSentence = {}
    for enti in entities:
        for menti in entities[enti]:
            key = menti[4] * 10 + menti[0]
            vertexSentence[key] = copy.deepcopy(menti)
    vertexlists = sorted(vertexSentence.items(), key=lambda item: item[0])
    vertexsentencelist = []
    for item in vertexlists:
        vertexsentencelist += sents[item[1][0]:item[1][1]]
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


def read_bio_test(args, file_in, tokenizer, max_seq_length=1024, save_file=''):
    # 缓存处理的数据
    if len(save_file) > 2 and os.path.exists(save_file):
        with open(file=save_file, mode='rb') as fr:
            features = pickle.load(fr)
            fr.close()
        print('load preprocessed data from {}.'.format(save_file))
        return features

    rel2id = None
    if args.task == 'cdr':
        rel2id = cdr_rel2id
    elif args.task == 'gda':
        rel2id = gda_rel2id
    assert rel2id is not None
    pmids = set()
    features = []
    maxlen = 0
    pos_samples = 0
    neg_samples = 0
    nlp = spacy.load('en_core_web_sm')
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            # todo debug
            # if len(features) < 10395:
            #     features.append([])
            #     continue

            line = line.rstrip().split('\t')
            pmid = line[0]
            # gnn 对应一个长宽均为实体类型数的矩阵，如果两实体在同一句子中，标记为0
            inter_mask = []
            entities = {}
            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                sents = [t.split(' ') for t in text.split('|')]
                sent_len = [len(i) for i in sents]
                sents_len = []
                for i in range(len(sent_len)):
                    if i == 0:
                        sents_len.append(sent_len[0])
                    else:
                        sents_len.append(sent_len[i] + sents_len[i - 1])
                prs = chunks(line[2:], 17)
                ent2idx = {}
                train_triples = {}
                entity_pos = set()
                for p in prs:
                    if p[0] == "not_include":
                        continue
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    entity_str = list(map(str, p[6].split('|')))
                    entity_id = p[5]
                    if entity_id not in entities:
                        entities[entity_id] = []
                    for start, end, string in zip(es, ed, entity_str):
                        entity_pos.add((start, end, tpy))
                        # 当前实体在哪个句子中
                        sent_in_id = -1
                        for i in range(len(sents_len)):
                            if sents_len[i] > end:
                                sent_in_id = i
                                break
                        if [start, end, tpy, string, sent_in_id] not in entities[entity_id]:
                            entities[entity_id].append([start, end, tpy, string, sent_in_id])

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    entity_str = list(map(str, p[12].split('|')))
                    entity_id = p[11]
                    if entity_id not in entities:
                        entities[entity_id] = []
                    for start, end, string in zip(es, ed, entity_str):
                        entity_pos.add((start, end, tpy))
                        # 当前实体在哪个句子中
                        sent_in_id = -1
                        for i in range(len(sents_len)):
                            if sents_len[i] > end:
                                sent_in_id = i
                                break
                        if [start, end, tpy, string, sent_in_id] not in entities[entity_id]:
                            entities[entity_id].append([start, end, tpy, string, sent_in_id])

                for i in range(0, len(entities)):
                    inter_mask.append([1] * len(entities))
                for i in range(0, len(entities)):
                    inter_mask[i][i] = 0
                for i, ent1 in enumerate(entities):
                    for j, ent2 in enumerate(entities):
                        if i != j:
                            breakFlag = 0
                            for men1 in entities[ent1]:
                                for men2 in entities[ent2]:
                                    if men1[4] == men2[4]:
                                        inter_mask[i][j] = 0
                                        inter_mask[j][i] = 0
                                        breakFlag = 1
                                        break
                                if breakFlag == 1:
                                    break

                # spacy 分析
                nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
                doc = nlp(text.replace('|', ' '))
                spacy_tokens = []
                spacy_offset = nlp("*")[0]
                for token in doc:
                    spacy_tokens.append(token)
                # 依据spacy的分词解析结果，存放开始的index
                # id对应的单词
                index2word = {}
                # 一个单词对应的所有分词片段id
                word2piecesid = {}
                # spacy中token的当前id
                spacy_token_id = 0

                # entitys 中缺少来源于哪个句子的id，可以考虑添加
                sents = [t.split(' ') for t in text.split('|')]
                # 定义了一种离散方式将实体分割成词放入list中，作为一个新的句子
                # vertexsentencelist = addEntitySentence(entities, sents)
                # sents.append(vertexsentencelist)

                new_sents = []
                token_map = []
                lengthofPice = 0
                sent_map = {}
                entity_pos = list(entity_pos)
                entity_pos.sort()
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        # 每个实体的起始位置都加进去
                        oneToken = []
                        eid = 0
                        for i, ep in enumerate(entity_pos):
                            start, end, tpy = ep
                            if i_t == start or i_t == end:
                                eid = i
                                break
                        if i_t == entity_pos[eid][0] or i_t == entity_pos[eid][1]:
                            # 目前认为标记实体的'*'应该算在是实体的一部分，但不是一个单词的一部分，故不算其中。
                            index2word[len(index2word)] = spacy_offset
                            for token_wordpiece in tokens_wordpiece:
                                index2word[len(index2word)] = spacy_tokens[spacy_token_id]
                                if spacy_tokens[spacy_token_id] not in word2piecesid:
                                    word2piecesid[spacy_tokens[spacy_token_id]] = []
                                word2piecesid[spacy_tokens[spacy_token_id]].append(len(index2word) - 1)

                            oneToken.append(lengthofPice + 1)
                            tokens_wordpiece = ["*"] + tokens_wordpiece
                            lengthofPice += len(tokens_wordpiece)
                            oneToken.append(lengthofPice)
                        else:
                            for token_wordpiece in tokens_wordpiece:
                                index2word[len(index2word)] = spacy_tokens[spacy_token_id]
                                if spacy_tokens[spacy_token_id] not in word2piecesid:
                                    word2piecesid[spacy_tokens[spacy_token_id]] = []
                                word2piecesid[spacy_tokens[spacy_token_id]].append(len(index2word) - 1)

                            oneToken.append(lengthofPice)
                            lengthofPice += len(tokens_wordpiece)
                            oneToken.append(lengthofPice)
                        # 相当于docred中的new_map，分词后每个词对应的位置
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        token_map.append(oneToken)
                        i_t += 1
                        spacy_token_id += 1
                    # sent_map[i_t] = len(new_sents)
                sents = new_sents
                # 长度限制 1024
                if len(new_sents) > 1022:
                    print(pmid, "too long")
                    continue

                entity_pos = []
                # todo debug
                # if len(sent_map) == 224 and len(features) > 2517:
                #     print()

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

                    r = rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                        if mention["relation"] != 0:
                            pos_samples += 1
                        else:
                            neg_samples += 1
                    relations.append(relation)
                    hts.append([h, t])

            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)

            a_mentions = np.eye(len(input_ids))
            # 句法依存树
            adj_syntactic_dependency_tree = np.eye(len(input_ids))

            offset = 1
            edges = 0
            for token_s in token_map:
                start = token_s[0] + offset
                end = token_s[1] + offset
                for i in range(start, end):
                    for j in range(start, end):
                        if i < (len(input_ids) - 1) and j < (len(input_ids) - 1):
                            if a_mentions[i][j] == 0:
                                a_mentions[i][j] = 1
                                edges += 1
            # 所有实体在 tokens 中的跨度
            mentionsofPice = []
            for eid in entities:
                for i in entities[eid]:
                    ment = [i[0], i[1]]
                    mentionsofPice.append([token_map[ment[0]][0], token_map[ment[1] - 1][1]])
            for ment in mentionsofPice:
                start = ment[0] + offset
                end = ment[1] + offset
                for i in range(start, end):
                    for j in range(start, end):
                        if i < (len(input_ids) - 1) and j < (len(input_ids) - 1):
                            if a_mentions[i][j] == 0:
                                a_mentions[i][j] = 1
                                edges += 1
            # 各类实体的实体跨度
            entityofPice = []
            for ent in entity_pos:
                # 每个单词（属于实体）的起始位置，可能是字母或者索引
                oneEntityP = []
                for ment in ent:
                    if (ment[0] + offset) == ment[1]:
                        oneEntityP.append(ment[0] + offset)
                    for i in range(ment[0] + offset, ment[1]):
                        oneEntityP.append(i)
                entityofPice.append(oneEntityP)
            predicted_Doc2 = []
            for h in range(0, len(entities)):
                item = [0, h]
                predicted_Doc2.append(item)

            predictedEntityPairPiece = []
            for item in predicted_Doc2:
                one_predicted = entityofPice[item[0]] + entityofPice[item[1]]
                predictedEntityPairPiece.append(one_predicted)
            for line in predictedEntityPairPiece:
                for i in line:
                    for j in line:
                        if a_mentions[i + offset][j + offset] == 0:
                            a_mentions[i + offset][j + offset] = 1
                            edges += 1
            for i in range(0, len(a_mentions)):
                a_mentions[i][i] = 1

            # 句法树
            count = 0
            i = 0
            while i < len(input_ids):
                if index2word[i] == spacy_offset:
                    i += 1
                    continue
                word = spacy_tokens[count]
                word_sp = tokenizer.tokenize(word.text)
                for child in word.children:
                    adj_word_list = word2piecesid[child]
                    word_list = word2piecesid[word]
                    # obtain the start index of child
                    child_key = next(key for key, val in index2word.items() if val == child)
                    # obtain the start index of spacy_word
                    word_key = next(key for key, val in index2word.items() if val == word)
                    # print("child:{}, word:{}".format(child, word))
                    for m in range(child_key, len(adj_word_list) + child_key):
                        for n in range(word_key, len(word_list) + word_key):
                            # print("m:{}, n:{}".format(m, n))
                            adj_syntactic_dependency_tree[m][n] = 1  # 无向图
                            adj_syntactic_dependency_tree[n][m] = 1

                i += len(word_sp)
                count += 1

            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
            a_mentions_new = [[0 for j in range(max_seq_length)] for i in range(max_seq_length)]
            adj_syntactic_dependency_tree_new = [[0 for j in range(max_seq_length)] for i in range(max_seq_length)]
            if args.transformer_type == "bert":
                for i in range(len(a_mentions)):
                    for j in range(len(a_mentions)):
                        a_mentions_new[i + 1][j + 1] = a_mentions[i][j]
                for i in range(len(adj_syntactic_dependency_tree)):
                    for j in range(len(adj_syntactic_dependency_tree)):
                        adj_syntactic_dependency_tree_new[i + 1][j + 1] = adj_syntactic_dependency_tree[i][j]
            elif args.transformer_type == "roberta":
                for i in range(len(a_mentions)):
                    for j in range(len(a_mentions)):
                        a_mentions_new[i + 1][j + 1] = a_mentions[i][j]
                for i in range(len(adj_syntactic_dependency_tree)):
                    for j in range(len(adj_syntactic_dependency_tree)):
                        adj_syntactic_dependency_tree_new[i + 1][j + 1] = adj_syntactic_dependency_tree[i][j]

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           'Adj': a_mentions_new,
                           'adj_syntactic_dependency_tree': adj_syntactic_dependency_tree_new,
                           'inter_mask': inter_mask
                           }
                features.append(feature)

            if args.demo == 'true' and len(features) > 100:
                break

    if len(save_file) > 2:
        with open(file=save_file, mode='wb') as fw:
            pickle.dump(features, fw)
        print('finish reading {} and save preprocessed data to {}.'.format(file_in, save_file))
    return features
