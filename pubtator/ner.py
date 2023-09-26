from tqdm import tqdm
import ujson as json
import numpy as np
import unidecode
from convert_pubtator import add_annotations_2_text_instances, PubtatorDocument
from transformers import AutoConfig, AutoModel, AutoTokenizer
import random
import os
import re
import nltk
from data_preproc.pubtator.document import PubtatorDocument, TextInstance
from data_preproc.pubtator.annotation import AnnotationInfo
from nltk import word_tokenize


def load_pubtator_into_documents(in_pubtator_file,
                                 normalized_type_dict={},
                                 re_id_spliter_str=r'\,',
                                 pmid_2_index_2_groupID_dict=None):
    documents = []

    with open(in_pubtator_file, 'r', encoding='utf8') as pub_reader:

        pmid = ''

        document = None

        annotations = []
        text_instances = []
        relation_pairs = {}
        index2normalized_id = {}
        id2index = {}

        for line in pub_reader:
            line = line.rstrip()

            if line == '':
                document = PubtatorDocument(pmid)
                # print(pmid)
                add_annotations_2_text_instances(text_instances, annotations)
                document.text_instances = text_instances
                document.relation_pairs = relation_pairs
                documents.append(document)

                annotations = []
                text_instances = []
                relation_pairs = {}
                id2index = {}
                index2normalized_id = {}
                continue

            tks = line.split('|')

            if len(tks) > 1 and (tks[1] == 't' or tks[1] == 'a'):
                # 2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                pmid = tks[0]
                x = TextInstance(tks[2])
                text_instances.append(x)
            else:
                _tks = line.split('\t')
                if len(_tks) == 6:
                    start = int(_tks[1])
                    end = int(_tks[2])
                    index = _tks[1] + '|' + _tks[2]
                    text = _tks[3]
                    ne_type = _tks[4]
                    ne_type = re.sub('\s*\(.*?\)\s*$', '', ne_type)
                    orig_ne_type = ne_type
                    if ne_type in normalized_type_dict:
                        ne_type = normalized_type_dict[ne_type]

                    _anno = AnnotationInfo(start, end - start, text, ne_type)

                    # 2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                    ids = [x for x in re.split(re_id_spliter_str, _tks[5])]

                    # if annotation has groupID then update its id
                    if orig_ne_type == 'SequenceVariant':
                        if pmid_2_index_2_groupID_dict != None and index in pmid_2_index_2_groupID_dict[pmid]:
                            index2normalized_id[index] = pmid_2_index_2_groupID_dict[pmid][index][
                                0]  # pmid_2_tmvarID_2_groupID_dict[pmid][_id] => (var_id, gene_id)
                            _anno.corresponding_gene_id = pmid_2_index_2_groupID_dict[pmid][index][1]
                    for i, _id in enumerate(ids):
                        if pmid_2_index_2_groupID_dict != None and index in pmid_2_index_2_groupID_dict[pmid]:
                            id2index[ids[i]] = index
                            ids[i] = pmid_2_index_2_groupID_dict[pmid][index][
                                0]  # pmid_2_tmvarID_2_groupID_dict[pmid][_id] => (var_id, gene_id)
                            _anno.corresponding_gene_id = pmid_2_index_2_groupID_dict[pmid][index][1]
                        else:
                            # ids[i] = re.sub('\s*\(.*?\)\s*$', '', _id)
                            ids[i] = _id

                    _anno.orig_ne_type = orig_ne_type
                    _anno.ids = set(ids)
                    annotations.append(_anno)
                elif len(_tks) == 4 or len(_tks) == 5:

                    id1 = _tks[2]
                    id2 = _tks[3]

                    if pmid_2_index_2_groupID_dict != None and (id1 in id2index) and (
                            id2index[id1] in index2normalized_id):
                        id1 = index2normalized_id[
                            id2index[id1]]  # pmid_2_tmvarID_2_groupID_dict[pmid][_id] => (var_id, gene_id)
                    if pmid_2_index_2_groupID_dict != None and (id2 in id2index) and (
                            id2index[id2] in index2normalized_id):
                        id2 = index2normalized_id[
                            id2index[id2]]  # pmid_2_tmvarID_2_groupID_dict[pmid][_id] => (var_id, gene_id)
                    # rel_type = _tks[1]
                    rel_type = _tks[4]
                    relation_pairs[(id1, id2)] = rel_type

        if len(text_instances) != 0:
            document = PubtatorDocument(pmid)
            add_annotations_2_text_instances(text_instances, annotations)
            document.text_instances = text_instances
            document.relation_pairs = relation_pairs
            documents.append(document)

    return documents


all_documents = load_pubtator_into_documents('/py_project/SSR-PU/dataset/BioRED_Subtask1/Train.PubTator',
                                             normalized_type_dict={}, re_id_spliter_str=r'\|')
data = all_documents

# {"text": "Port conditions update - Syria - Lloyds Shipping . Port conditions from Lloyds Shipping Intelligence Service -- LATTAKIA , Aug 10 - waiting time at Lattakia and Tartous presently 24 hours .", "entity_types": ["LOC", "ORG", "ORG", "LOC", "LOC", "LOC"], "entity_start_chars": [25, 33, 72, 112, 148, 161], "entity_end_chars": [30, 48, 108, 120, 156, 168], "id": "train_09", "word_start_chars": [0, 5, 16, 23, 25, 31, 33, 40, 49, 51, 56, 67, 72, 79, 88, 101, 109, 112, 121, 123, 127, 130, 132, 140, 145, 148, 157, 161, 169, 179, 182, 188], "word_end_chars": [4, 15, 22, 24, 30, 32, 39, 48, 50, 55, 66, 71, 78, 87, 100, 108, 111, 120, 122, 126, 129, 131, 139, 144, 147, 156, 160, 168, 178, 181, 187, 189]}

for idx, sample in tqdm(enumerate(data), desc="Documents"):
    text = sample.text_instances[0].text + " " + sample.text_instances[1].text
    text = unidecode.unidecode(text)
    words = word_tokenize(text)
    id = sample.id
    entities = sample.text_instances[0].annotations + sample.text_instances[1].annotations
    entity_types = []
    entity_start_chars = []
    entity_end_chars = []
    word_start_chars = []
    word_end_chars = []
    for entity in entities:
        entity_types.append(entity.ne_type)
        entity_start_chars.append(entity.position)
        entity_end_chars.append(entity.position + entity.length)
    start_char_id = 0
    end_char_id = 0
    for token in tokens:
        start_char_id = end_char_id + 1
        word_start_chars.append(token)
        word_end_chars.append()

    print()
