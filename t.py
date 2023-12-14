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

text = "4 - diphenylacetoxy - N - methylpiperidine"
nlp = spacy.load('en_core_web_sm')
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
# todo 这里可以尝试使用np.zero
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
        print("child:{}, word:{}".format(child, word))
        for m in range(child_key, len(adj_word_list) + child_key):
            for n in range(word_key, len(word_list) + word_key):
                print("m:{}, n:{}".format(m, n))
                adj_matrix[m][n] = 1  # 无向图
                adj_matrix[n][m] = 1

    i += len(word_sp)
    count += 1

print(adj_matrix)

# 依存句法树打印输出
displacy.serve(doc, style='dep')

print()
