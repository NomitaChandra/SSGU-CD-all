#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/08/2019

author: fenia
"""

import os, re, sys
from utils import replace2symbol, replace2space
from collections import OrderedDict
from tqdm import tqdm
from recordtype import recordtype

TextStruct = recordtype('TextStruct', 'pmid txt')
EntStruct = recordtype('EntStruct', 'pmid name off1 off2 type kb_id sent_no word_id bio')
RelStruct = recordtype('RelStruct', 'pmid type arg1 arg2')
PairStruct = recordtype('PairStruct', 'pmid type arg1 arg2 dir cross closest')


def readPubTator(args, split=';', biored_cd=False):
    """
    Read data and store in structs
    """
    if not os.path.exists('/'.join(args.output_file.split('/')[:-1])):
        os.makedirs('/'.join(args.output_file.split('/')[:-1]))

    abstracts = OrderedDict()
    entities = OrderedDict()
    relations = OrderedDict()
    id2type = {}
    rel_types = []

    with open(args.input_file, 'r') as infile:
        for line in tqdm(infile):
            # text
            if len(line.rstrip().split('|')) == 3 and \
                    (line.strip().split('|')[1] == 't' or line.strip().split('|')[1] == 'a'):
                line = line.strip().split('|')

                pmid = line[0]
                text = line[2]
                if text[-1] != '.' and text[-1] != '!' and text[-1] != '?' and line[1] == 'a':
                    text = text + '.'

                # replace weird symbols and spaces
                text = replace2symbol(text)
                text = replace2space(text)

                if pmid not in abstracts:
                    abstracts[pmid] = [TextStruct(pmid, text)]
                else:
                    abstracts[pmid] += [TextStruct(pmid, text)]

            # entities
            elif len(line.rstrip().split('\t')) == 6:
                line = line.strip().split('\t')
                pmid = line[0]
                offset1 = int(line[1])
                offset2 = int(line[2])
                ent_name = line[3]
                ent_type = line[4]
                kb_id = line[5].split(split)
                for kb in kb_id:
                    id2type[kb] = ent_type

                if biored_cd and ent_type != 'ChemicalEntity' and ent_type != 'DiseaseOrPhenotypicFeature':
                    continue

                # replace weird symbols and spaces
                ent_name = replace2symbol(ent_name)
                ent_name = replace2space(ent_name)

                # currently consider each possible ID as another entity
                for k in kb_id:
                    if pmid not in entities:
                        entities[pmid] = [EntStruct(pmid, ent_name, offset1, offset2, ent_type, [k], -1, [], [])]
                    else:
                        entities[pmid] += [EntStruct(pmid, ent_name, offset1, offset2, ent_type, [k], -1, [], [])]

            elif len(line.rstrip().split('\t')) == 7:
                line = line.strip().split('\t')
                pmid = line[0]
                offset1 = int(line[1])
                offset2 = int(line[2])
                ent_name = line[3]
                ent_type = line[4]
                kb_id = line[5].split(split)
                extra_ents = line[6].split(split)
                for kb in kb_id:
                    id2type[kb] = ent_type

                if biored_cd and ent_type != 'ChemicalEntity' and ent_type != 'DiseaseOrPhenotypicFeature':
                    continue

                # replace weird symbols and spaces
                ent_name = replace2symbol(ent_name)
                ent_name = replace2space(ent_name)
                for i, e in enumerate(extra_ents):
                    if pmid not in entities:
                        entities[pmid] = [EntStruct(pmid, ent_name, offset1, offset2, ent_type, [kb_id[i]], -1, [], [])]
                    else:
                        entities[pmid] += [
                            EntStruct(pmid, ent_name, offset1, offset2, ent_type, [kb_id[i]], -1, [], [])]

            # relations
            elif len(line.rstrip().split('\t')) == 4:
                line = line.strip().split('\t')
                pmid = line[0]
                rel_type = line[1]
                arg1 = tuple((line[2].split(split)))
                arg2 = tuple((line[3].split(split)))

                if pmid not in relations:
                    relations[pmid] = [RelStruct(pmid, rel_type, arg1, arg2)]
                else:
                    relations[pmid] += [RelStruct(pmid, rel_type, arg1, arg2)]

            # BioRED relations
            elif len(line.rstrip().split('\t')) == 5:
                line = line.strip().split('\t')
                pmid = line[0]
                rel_type = line[1]
                arg1 = tuple((line[2].split(split)))
                arg2 = tuple((line[3].split(split)))
                ent_type1 = id2type[line[2]]
                ent_type2 = id2type[line[3]]

                if biored_cd and ((ent_type1 != 'ChemicalEntity' and ent_type1 != 'DiseaseOrPhenotypicFeature') or (
                        ent_type2 != 'ChemicalEntity' and ent_type2 != 'DiseaseOrPhenotypicFeature')):
                    continue
                if ent_type1 == ent_type2:
                    continue

                if arg1 == arg2:
                    continue

                if pmid not in relations:
                    relations[pmid] = [RelStruct(pmid, rel_type, arg1, arg2)]
                else:
                    relations[pmid] += [RelStruct(pmid, rel_type, arg1, arg2)]
                if rel_type not in rel_types:
                    rel_types.append(rel_type)
                    print(pmid, arg1, arg2)

            elif line == '\n':
                continue
    print(rel_types)
    return abstracts, entities, relations
