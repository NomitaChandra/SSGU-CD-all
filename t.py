import spacy
import torch
import numpy as np
from spacy import displacy
from transformers import BertTokenizerFast
from transformers import AutoConfig, AutoModel, AutoTokenizer
from model_utils.tree import Tree, head_to_tree, tree_to_adj

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

# APLLoss_devf1 = (85.71378563955724 + 86.1035923674988 + 85.91499226027506 + 86.04483937933473) / 4
# APLLoss_testf1 = (84.95436613943848 + 85.23406845833507 + 85.24863715859198 + 85.29071474483379) / 4
#
# ATLoss_devf1 = (85.61423287683874 + 85.82744546559776 + 85.84197176956602 + 85.9985026695822) / 4
# ATLoss_testf1 = (84.99950018651245 + 84.88664449266203 + 84.85692412472044 + 85.09541771495479) / 4
#
# BalancedLoss_devf1 = (86.29806823194548 + 86.41171331459596 + 86.58486538146019 + 86.41420857493209) / 4
# BalancedLoss_testf1 = (85.18836773390358 + 85.24702544845914 + 85.44597312111769 + 85.31696112937529) / 4
#
# CrossEntropyLoss_devf1 = (86.49675876823754 + 86.57566821626959 + 86.45887738327612 + 86.70296137118333) / 4
# CrossEntropyLoss_testf1 = (86.85441711285755 + 86.247044471604 + 86.58128269225624 + 86.49925478209825) / 4
#
# print()


text = "During an 18 - month period of study 41 hemodialyzed patients receiving desferrioxamine ( 10 - 40 mg / kg BW / 3 times weekly ) for the first time were monitored for detection of audiovisual toxicity ."
nlp = spacy.load('en_core_web_lg')
tokenizer = AutoTokenizer.from_pretrained('/home/yjs1217/Downloads/pretrained/scibert_scivocab_cased')

sent = text.split(" ")

new_sent = []
spacy_tokens = []
doc = nlp(text)
for token in doc:
    spacy_tokens.append(token)
for i in sent:
    new_sent.extend(tokenizer.tokenize(i))

sen_tokens_text_list = new_sent

# 依据spacy的分词解析结果，存放开始的index
index2word = {}
word2sttlid = {}
last_index = 0
sttlid = 0
for word in spacy_tokens:
    index = text.index(str(word), last_index)
    index2word[sttlid] = word
    word_sp = tokenizer.tokenize(word.text)
    word2sttlid[word] = [sttlid + i for i in range(len(word_sp))]
    sttlid += len(word_sp)
    last_index = index + len(word)

count = 0
adj_matrix = np.eye(len(sen_tokens_text_list))
i = 0
while i < len(sen_tokens_text_list):
    word = spacy_tokens[count]
    word_sp = tokenizer.tokenize(word.text)
    for child in word.children:
        adj_word_list = word2sttlid[child]
        word_list = word2sttlid[word]
        child_key = next(key for key, val in index2word.items() if val == child)  # obtain the start index of child
        word_key = next(key for key, val in index2word.items() if val == word)  # obtain the start index of spacy_word
        # print("child:{}, word:{}".format(child, word))
        for m in range(child_key, len(adj_word_list) + child_key):
            for n in range(word_key, len(word_list) + word_key):
                # print("m:{}, n:{}".format(m, n))
                adj_matrix[m][n] = 1  # 无向图
                adj_matrix[n][m] = 1

    i += len(word_sp)
    count += 1

print(adj_matrix)

# 依存句法树打印输出
displacy.serve(doc, style='dep')

print()
