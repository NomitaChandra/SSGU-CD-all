import spacy
import torch
import numpy as np
from spacy import displacy
from transformers import BertTokenizerFast
from transformers import AutoConfig, AutoModel, AutoTokenizer

# pred = torch.tensor([[1.0000, 0.2000, 0.1000, 0.5000],
#                      [0.3000, 0.4000, 0.7000, 0.6000],
#                      [0.6000, 0.5000, 0.2000, 0.9000]])
#
# pred = pred[..., :1]
#
# # 创建一个二维张量
# tensor = torch.tensor([[10, 20], [30, 40], [50, 60], [70, 80]])
#
# # 在第一个维度上获取最大的2个元素
# values, indices = torch.topk(tensor, 2, dim=0)
#
# print("Values:", values)  # 输出最大的2个值
# print("Indices:", indices)  # 输出这些值的原始索引

# ours
test_F1 = [85.17736619751594, 85.332833889455, 85.38717488897824, 85.31696112937529, 86.13959031315775,
           86.98145995402288, 87.14375928888524, 86.61678449401478]
dev_F1 = [86.20465350846499, 86.37689592491846, 86.16270270734991, 86.19069819958453, 86.7233761927871,
          86.82197554020857, 86.83898073267308, 86.51861398456442]
print(sum(test_F1) / len(test_F1))
print(sum(dev_F1) / len(dev_F1))
# ba
test_F1 = [85.18836773390358, 85.24702544845914, 85.44597312111769, 85.31696112937529, 86.4301787797973,
           86.61367358965704, 86.48865233861675, 86.45937237221595, 85.89693636814066, 85.98173151982502,
           85.78454281939337, 85.98173151982502]
dev_F1 = [86.29806823194548, 86.41171331459596, 86.58486538146019, 86.41420857493209, 86.94281925597403,
          86.88557538015306, 86.92959062831807, 86.84160468210304, 85.9306735172644, 85.87291716259361,
          85.81546294625232, 85.84424835181083]
print(sum(test_F1) / len(test_F1))
print(sum(dev_F1) / len(dev_F1))
# ATL
test_F1 = [84.99950018651245, 84.88664449266203, 84.85692412472044, 85.09541771495479, 84.85396523354063,
           84.92288173204825, 84.75197594842716, 84.83900673385544]
dev_F1 = [85.61423287683874, 85.82744546559776, 85.84197176956602, 85.9985026695822, 85.27713588791383,
          85.2906130808291, 85.24707497658865, 85.24707497658865]
print(sum(test_F1) / len(test_F1))
print(sum(dev_F1) / len(dev_F1))
# ASL
test_F1 = [85.00204459892741, 84.8695996348086, 85.12095974756922, 85.32884250851346, 85.42764364562969,
           84.96682199445362, 85.28478668347944, 86.4729295676365, 86.42045027434482, 86.50435414276416,
           86.47809906664624]
dev_F1 = [84.81295357651906, 84.88322246960614, 85.07021857936637, 85.30415928837371, 85.05104645526123,
          85.16547546283891, 85.18468515847361, 86.94729042359282, 87.04403382984502, 86.98684120190904,
          87.1060984789924]
print(sum(test_F1) / len(test_F1))
print(sum(dev_F1) / len(dev_F1))
# APL
test_F1 = [84.9404452597257, 84.96746496841887, 84.7604572407089, 84.81781860369182, 84.95436613943848,
           85.23406845833507, 85.24863715859198, 85.29071474483379]
dev_F1 = [85.64104745895543, 85.52379642601616, 85.20783391567717, 85.53540062509028, 85.71378563955724,
          86.1035923674988, 85.91499226027506, 86.04483937933473]
print(sum(test_F1) / len(test_F1))
print(sum(dev_F1) / len(dev_F1))
# text = "During an 18 - month period of study 41 hemodialyzed patients receiving desferrioxamine ( 10 - 40 mg / kg BW / 3 times weekly ) for the first time were monitored for detection of audiovisual toxicity ."
# nlp = spacy.load('en_core_web_lg')
# tokenizer = AutoTokenizer.from_pretrained('/home/yjs1217/Downloads/pretrained/scibert_scivocab_cased')
#
# sent = text.split(" ")
#
# new_sent = []
# spacy_tokens = []
# doc = nlp(text)
# for token in doc:
#     spacy_tokens.append(token)
# for i in sent:
#     new_sent.extend(tokenizer.tokenize(i))
#
# sen_tokens_text_list = new_sent
#
# # 依据spacy的分词解析结果，存放开始的index
# index2word = {}
# word2sttlid = {}
# last_index = 0
# sttlid = 0
# for word in spacy_tokens:
#     index = text.index(str(word), last_index)
#     index2word[sttlid] = word
#     word_sp = tokenizer.tokenize(word.text)
#     word2sttlid[word] = [sttlid + i for i in range(len(word_sp))]
#     sttlid += len(word_sp)
#     last_index = index + len(word)
#
# count = 0
# adj_matrix = np.eye(len(sen_tokens_text_list))
# i = 0
# while i < len(sen_tokens_text_list):
#     word = spacy_tokens[count]
#     word_sp = tokenizer.tokenize(word.text)
#     for child in word.children:
#         adj_word_list = word2sttlid[child]
#         word_list = word2sttlid[word]
#         child_key = next(key for key, val in index2word.items() if val == child)  # obtain the start index of child
#         word_key = next(key for key, val in index2word.items() if val == word)  # obtain the start index of spacy_word
#         # print("child:{}, word:{}".format(child, word))
#         for m in range(child_key, len(adj_word_list) + child_key):
#             for n in range(word_key, len(word_list) + word_key):
#                 # print("m:{}, n:{}".format(m, n))
#                 adj_matrix[m][n] = 1  # 无向图
#                 adj_matrix[n][m] = 1
#
#     i += len(word_sp)
#     count += 1
#
# print(adj_matrix)
#
# # 依存句法树打印输出
# displacy.serve(doc, style='dep')
#
# print()
