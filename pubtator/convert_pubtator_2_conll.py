import json

import unidecode
from convert_pubtator import add_annotations_2_text_instances
import re
from document import PubtatorDocument, TextInstance
from annotation import AnnotationInfo
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


def get_ner_dataset_biored(filepath):
    # filepath='data_preproc/biored/Train.PubTator'
    all_documents = load_pubtator_into_documents(filepath, normalized_type_dict={}, re_id_spliter_str=r'\|')
    data = all_documents
    ner_documents = []
    types = []
    for idx, sample in enumerate(data):
        text = "" + sample.text_instances[0].text + " " + sample.text_instances[1].text
        text = unidecode.unidecode(text)
        id = sample.id
        for annotation in sample.text_instances[1].annotations:
            annotation.position = annotation.position + sample.text_instances[1].offset
        entities = sample.text_instances[0].annotations + sample.text_instances[1].annotations
        offset = 0
        for entity in entities:
            entity_position = entity.position
            if entity_position + offset != 0 and text[entity_position + offset - 1] != " ":
                text = text[0:entity_position + offset] + " " + text[entity_position + offset:]
                offset += 1
            entity.position = entity.position + offset
            if entity_position + entity.length + offset <= len(text) and text[
                entity_position + entity.length + offset] != " ":
                # print(text[entity.position + entity.length + offset])
                text = text[0:entity_position + entity.length + offset] + " " + text[
                                                                                entity_position + entity.length + offset:]
                offset += 1
        words = word_tokenize(text)

        entity_types = []
        entity_start_chars = []
        entity_end_chars = []
        word_start_chars = []
        word_end_chars = []
        start_char_id = 0
        end_char_id = 0
        entity_id = 0
        for word in words:
            start_char_id = text[end_char_id:].find(word) + end_char_id
            end_char_id = start_char_id + len(word)
            word_start_chars.append(start_char_id)
            word_end_chars.append(end_char_id)
            if entity_id >= len(entities):
                continue
            entity = entities[entity_id]
            entity_start_id = entity.position
            entity_end_id = entity.position + entity.length

            if start_char_id == entity_start_id:
                entity_start_chars.append(entity_start_id)
            if entity_end_id == end_char_id:
                entity_end_chars.append(entity_end_id)
                entity_types.append("" + entity.ne_type)
                entity_id += 1

        # print(text)
        # print(words)
        # print(word_start_chars)
        # print(word_end_chars)
        # print(entity_start_chars)
        # print(entity_end_chars)
        # print()
        if len(word_start_chars) != len(word_end_chars) or len(word_start_chars) != len(words):
            print('error')
        if len(entity_start_chars) != len(entity_end_chars) or len(entity_start_chars) != len(entities):
            print('error')
        if len(word_tokenize(text)) != len(word_start_chars):
            print(len(word_tokenize(text)), len(word_start_chars), 'error tokenize')
        use1 = word_tokenize(text)
        use2 = word_start_chars
        use3 = word_end_chars
        for i in range(len(use1)):
            if use2[i] in entity_start_chars:
                idex = entity_start_chars.index(use2[i])
                # print(entity_types[idex], entity_start_chars[idex], entity_end_chars[idex])
                if entity_end_chars[idex] not in use3:
                    print('e')
            if use3[i] - use2[i] != len(use1[i]):
                print('e')
            # print(i, use1[i], use2[i], use3[i])

        ner_document = json.dumps({"text": text, "entity_types": entity_types,
                                   "entity_start_chars": entity_start_chars, "entity_end_chars": entity_end_chars,
                                   "id": id, "word_start_chars": word_start_chars, "word_end_chars": word_end_chars})

        ner_documents.append(ner_document)
    # print(types)
    return ner_documents


def main():
    documents = get_ner_dataset_biored('data_preproc/biored/Train.PubTator')
    with open('data/biored/train.json', "w") as writer:
        for document in documents:
            writer.write(str(document) + '\n')
    documents = get_ner_dataset_biored('data_preproc/biored/Dev.PubTator')
    with open('data/biored/dev.json', "w") as writer:
        for document in documents:
            writer.write(str(document) + '\n')
    documents = get_ner_dataset_biored('data_preproc/biored/Test.PubTator')
    with open('data/biored/test.json', "w") as writer:
        for document in documents:
            writer.write(str(document) + '\n')


if __name__ == "__main__":
    main()
