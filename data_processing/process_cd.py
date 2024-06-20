from recordtype import recordtype
import argparse
from readers import *

TextStruct = recordtype('TextStruct', 'pmid txt')
EntStruct = recordtype('EntStruct', 'pmid name off1 off2 type kb_id sent_no word_id bio')
RelStruct = recordtype('RelStruct', 'pmid type arg1 arg2')


def main():
    """ 
    Main processing function 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str)
    parser.add_argument('--output_file', '-o', type=str)
    args = parser.parse_args()
    args.data = 'biored'
    abstracts, entities, relations = readPubTator(args, ';', biored_cd=True)

    with open(args.output_file, 'w') as data_out:
        for pmid in abstracts:
            data_out.write(pmid + '|t|' + abstracts[pmid][0].txt + '\n')
            data_out.write(pmid + '|a|' + abstracts[pmid][1].txt + '\n')
            if pmid in entities:
                for entity in entities[pmid]:
                    data_out.write(str(pmid) + '\t' + str(entity.off1) + '\t' + str(entity.off2) + '\t' + str(entity.name)
                                   + '\t' + str(entity.type) + '\t' + str(entity.kb_id) + '\n')
            if pmid in relations:
                for relation in relations[pmid]:
                    data_out.write(str(pmid) + '\t' + str(relation.type)
                                   + '\t' + str(relation.arg1[0]) + '\t' + str(relation.arg2[0]) + '\n')
            data_out.write('\n')


if __name__ == "__main__":
    main()
