import re
from pubtator.document import PubtatorDocument, TextInstance
from pubtator.annotation import AnnotationInfo
import os
import random


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
                # document = PubtatorDocument(pmid)
                # # print(pmid)
                # add_annotations_2_text_instances(text_instances, annotations)
                # document.text_instances = text_instances

                document = {'id': pmid, 'passages': [], 'relations': []}
                annotation_list = []
                for annotation in annotations:
                    ids = ''
                    for id in annotation.ids:
                        if len(ids) > 0:
                            ids += ','
                        ids = ids + id
                    annotation_list.append({
                        'infons': {'identifier': ids, 'type': annotation.ne_type},
                        'text': annotation.text,
                        'locations': [{'offset': annotation.position, 'length': annotation.length}]
                    })
                document['passages'].append(
                    {'offset': 0, 'text': text_instances[0].text, 'annotations': []}
                )
                document['passages'].append(
                    {'offset': len(text_instances[0].text) + 1, 'text': text_instances[1].text,
                     'annotations': annotation_list}
                )

                for relation in relation_pairs:
                    document['relations'].append(
                        {'infons': {'entity1': relation[0], 'entity2': relation[1], 'type': relation_pairs[relation]}}
                    )
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
                if len(_tks) >= 6:
                    # 过滤错误
                    if _tks[5] == '-1':
                        continue
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
                    ids = [x.strip('*') for x in re.split(re_id_spliter_str, _tks[5])]

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
                    rel_type = _tks[1]
                    relation_pairs[(id1, id2)] = rel_type

        if len(text_instances) != 0:
            document = {'id': pmid, 'passages': [], 'relations': []}
            annotation_list = []
            for annotation in annotations:
                ids = ''
                for id in annotation.ids:
                    if len(ids) > 0:
                        ids += ','
                    ids = ids + id
                annotation_list.append({
                    'infons': {'identifier': ids, 'type': annotation.ne_type},
                    'text': annotation.text,
                    'locations': [{'offset': annotation.position, 'length': annotation.length}]
                })
            document['passages'].append(
                {'offset': 0, 'text': text_instances[0].text, 'annotations': []}
            )
            document['passages'].append(
                {'offset': len(text_instances[0].text) + 1, 'text': text_instances[1].text,
                 'annotations': annotation_list}
            )

            for relation in relation_pairs:
                document['relations'].append(
                    {'infons': {'entity1': relation[0], 'entity2': relation[1], 'type': relation_pairs[relation]}}
                )
            documents.append(document)

    return documents


def add_annotations_2_text_instances(text_instances, annotations):
    offset = 0
    for text_instance in text_instances:
        text_instance.offset = offset
        offset += len(text_instance.text) + 1

    for annotation in annotations:
        can_be_mapped_to_text_instance = False

        for i, text_instance in enumerate(text_instances):
            if text_instance.offset <= annotation.position and annotation.position + annotation.length <= text_instance.offset + len(
                    text_instance.text):
                annotation.position = annotation.position - text_instance.offset
                text_instance.annotations.append(annotation)
                can_be_mapped_to_text_instance = True
                break
        if not can_be_mapped_to_text_instance:
            print(annotation.text)
            print(annotation.position)
            print(annotation.length)
            print(annotation, 'cannot be mapped to original text')
            raise

# def tokenize_documents_by_spacy(documents, spacy_model):
#     nlp = spacy.load(spacy_model)
#
#     for document in documents:
#         split_sentence(document, nlp)
#         tokenize_document_by_spacy(document, nlp)
