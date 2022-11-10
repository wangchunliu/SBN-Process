# coding-utf-8
import argparse
import re
import sys
import nltk
import random

nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from sbn_utils import get_sbn, sbn_string_to_list, list_to_file, file_to_list, is_number, between_quotes, is_operator, \
    is_role, word_level_sbn, char_level_sbn


def create_arg_parser():
    '''Create argument parser'''
    parser = argparse.ArgumentParser()
    # Parameter that are import to set
    parser.add_argument('-input_src', default='', type=str, help="SBN input-file")
    parser.add_argument('-input_tgt', default='', type=str, help="SBN output-file")
    parser.add_argument('-text_type', default="seq", type=str,
                        help="SBN data for seq2seq or graph model, options are [seq|graph]")
    parser.add_argument('-if_hyper', default="nohyper", type=str,
                        help="add hyper for seq2seq format data, options are [nohyper|hyper]")
    parser.add_argument('-if_anony', default="normal",
                        help="use anonymization to Named entities, options are [normal|anony]")
    parser.add_argument('-replace', default='', type=str,
                        help="replace unknown WordNet with concepts in training data, this is th path of training data")
    parser.add_argument('-trainfile', default='', type=str,
                        help="add hypernmy WordNet with concepts in training data, this is th path of training data")
    # parser.add_argument('-text_level', default='word', type=str, help="char-level only change number and named entiies, options are [word|char]")
    args = parser.parse_args()
    return args


def get_anonymize(new_clauses):
    return_strings = []
    alignment = {}
    random_list = [i for i in range(30)]
    random.shuffle(random_list)
    for index_clause, cur_clause in enumerate(new_clauses):
        for idx, item in enumerate(cur_clause):
            if (cur_clause[idx - 1] == 'Name' or cur_clause[idx - 1] == 'EQU') and between_quotes(cur_clause[idx]):
                cur_clause[idx] = "NAMEname" + str(random_list[index_clause % 30])
                ### for alignment output file
                alignment[cur_clause[idx]] = item.strip('"')  ###(1)
            if (cur_clause[idx - 1] == 'Name' or cur_clause[idx - 1] == 'EQU') and cur_clause[idx].startswith(
                    '"') and not cur_clause[idx].endswith('"'):
                cur_name = []
                for j in range(idx + 1, len(cur_clause), 1):
                    if cur_clause[j].endswith('"'):
                        end_index = j
                        for index in range(idx, end_index + 1):
                            cur_name.append(cur_clause[index].strip('"'))
                ### for alignment output file
                cur_clause[idx] = "NAMEname" + str(random_list[index_clause % 30])
                alignment[cur_clause[idx]] = ' '.join(cur_name)  ###(3)
                del cur_clause[idx + 1: end_index + 1]
        return_strings.extend([cur_clause])
    return return_strings, alignment


def split_sbn_list(list):
    cur_list = []
    all_list = []
    for i, item in enumerate(list):
        if item == "***":
            all_list.append(cur_list)
            cur_list = []
        else:
            cur_list.append(item.strip())
    all_list.append(cur_list)
    return all_list


def get_wordnet(train_file):
    sbns = get_sbn(train_file)
    train_concept_n = []
    train_concept_a = []
    train_concept_v = []
    train_concept_r = []
    for sbn in sbns:
        sbn = sbn_string_to_list(sbn)
        ### get sequence sbn data from raw sbn data
        for cur_sbn in sbn:
            # { Part-of-speech constants: ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v" }
            n = re.search(r'\.n\.', cur_sbn[0])
            a = re.search(r'\.a\.|\.s\.', cur_sbn[0])
            v = re.search(r'\.v\.', cur_sbn[0])
            r = re.search(r'\.r\.', cur_sbn[0])
            if n and cur_sbn[0] not in train_concept_n:
                train_concept_n.append(cur_sbn[0])
            if a and cur_sbn[0] not in train_concept_a:
                train_concept_a.append(cur_sbn[0])
            if v and cur_sbn[0] not in train_concept_v:
                train_concept_v.append(cur_sbn[0])
            if r and cur_sbn[0] not in train_concept_r:
                train_concept_r.append(cur_sbn[0])
    return train_concept_n, train_concept_a, train_concept_v, train_concept_r


def find_similar_nv(test_concept, train_concept):
    mostsimilarity_simi = float(0)
    mostsimilarity_hypo = float(0)
    mostsimilarity_hyper = float(0)
    # concept should be a right format of Wordnet
    print("Curent unknown wordnet: {0}".format(test_concept))
    try:
        wordnet_nltk = wn.synset(test_concept)  # adjective wordnet can use similarity
        if str(wordnet_nltk)[8:-2] in train_concept:
            return str(wordnet_nltk)[8:-2]
    except:
        print('Wordnet Error:{0}'.format(test_concept))
    else:
        # 1. 下义词 2.上义词
        all_hyponyms_wordnet = list(set([i for i in wordnet_nltk.closure(lambda s: s.hyponyms())]))
        all_hypernyms_wordnet = list(set([i for i in wordnet_nltk.closure(lambda s: s.hypernyms())]))
        all_wordnet = all_hyponyms_wordnet + all_hypernyms_wordnet
        if len(all_wordnet) > 0:
            try:
                for hyperhypo_wordnet in all_wordnet:
                    if str(hyperhypo_wordnet)[8:-2] in train_concept:
                        similarity = wordnet_nltk.path_similarity(hyperhypo_wordnet)
                        if float(similarity) > float(mostsimilarity_hyper):
                            mostsimilarity_hyper = similarity
                            similar_wordnet = str(hyperhypo_wordnet)[8:-2]
                return similar_wordnet
            except:
                print(('No hypernmy and hyponmy in training data: {0}'.format(test_concept)))
        # 3. 相似词
        try:
            for one_concept in train_concept:
                one_wordnet = wn.synset(one_concept)
                similarity = wordnet_nltk.path_similarity(one_wordnet)
                if similarity and float(similarity) > float(mostsimilarity_simi):
                    mostsimilarity_simi = similarity
                    similar_wordnet = one_concept
            return similar_wordnet
        except:
            print(('No similar concept in training data: {0}'.format(test_concept)))
    return test_concept


def find_similar_as(test_concept, train_concept):
    print("Curent unknown wordnet: {0}".format(test_concept))
    try:  # concept should be a right format of Wordnet
        wordnet_nltk = wn.synset(test_concept)  # adjective wordnet can use similarity
        if str(wordnet_nltk)[8:-2] in train_concept:
            return str(wordnet_nltk)[8:-2]
    except:
        print('Wordnet Error:{0}'.format(test_concept))
    else:
        # 1. also_sees() # 2. similar_tos()
        also_sees_wordnet = wordnet_nltk.also_sees()
        similar_tos_wordnet = wordnet_nltk.similar_tos()
        similar_wordnets = also_sees_wordnet + similar_tos_wordnet
        if len(similar_wordnets) > 0:
            try:
                for see_wordnet in similar_wordnets:
                    if str(see_wordnet)[8:-2] in train_concept:
                        return str(see_wordnet)[8:-2]
            except:
                print(('No Similar Wordnets in training data: {0}'.format(test_concept)))
        else:
            print(('No Similar Wordnets: {0}'.format(test_concept)))
    return test_concept


def add_hyper2seq(sbn_list):
    add_nodes_sbn = []
    for i, item in enumerate(sbn_list):
        add_nodes_sbn.append(item)
        if re.search(r'\.n\.', item) and "time.n.08" not in item and "entity.n.08" not in item:
            try:
                wordnet_nltk = wn.synset(item)
                hyper = lambda s: s.hypernyms()
                all_hyper = list(wordnet_nltk.closure(hyper))
                # print(all_hyper)
                for j, hyperhypo_wordnet in enumerate(all_hyper):
                    if hyperhypo_wordnet.name() in train_concept_n:
                        add_nodes_sbn.append('Hyper')
                        add_nodes_sbn.append(hyperhypo_wordnet.name())
                        break
                    else:
                        continue
            except:
                pass
        else:
            continue
    return add_nodes_sbn


if __name__ == "__main__":
    args = create_arg_parser()
    sbns = get_sbn(args.input_src)  ### get sbn list
    sen_list = file_to_list(args.input_tgt)  ### get sentence list
    outfile1 = open(args.input_src + ".{0}.{1}.{2}".format(args.text_type, args.if_anony, args.if_hyper), 'w',
                    encoding="utf-8")
    train_concept_n, train_concept_a, train_concept_v, train_concept_r = get_wordnet(args.trainfile)
    if args.if_anony == 'anony':
        outfile2 = open(args.input_tgt + ".{0}".format(args.if_anony), 'w', encoding="utf-8")
        outfile3 = open(args.input_src + '.alignment', 'w')
    for i, sbn in enumerate(sbns):
        sbn = sbn_string_to_list(sbn)
        if not args.if_anony == 'anony':
            if args.text_type == "seq":
                sbn_list_word = word_level_sbn(sbn)
                sbn_seq = [x for y in sbn_list_word for x in y]
                if args.if_hyper == "nohyper":
                    outfile1.write(' '.join(sbn_seq) + '\n')
                if args.if_hyper == "hyper":
                    sbn_seq = add_hyper2seq(sbn_seq)
                    outfile1.write(' '.join(sbn_seq) + '\n')
            else:
                sbn_graph = [' '.join(y) for y in sbn]
                outfile1.write(' *** '.join(sbn_graph) + '\n')
        ### get the anonymized sbn file
        else:
            sbn_list_word, alignment = get_anonymize(sbn)
            if args.text_type == "seq":
                sbn_seq = [x for y in sbn_list_word for x in y]
                outfile1.write(' '.join(sbn_seq) + '\n')
            else:
                sbn_graph = [' '.join(y) for y in sbn_list_word]
                outfile1.write(' *** '.join(sbn_graph) + '\n')
            ### get the anonymized sentence file
            sen = sen_list[i]
            for key, value in alignment.items():
                try:  ### two methods to get anonymized sentences
                    sen = re.sub(value, key, sen)
                except:
                    sen = sen.replace(value, key)
            outfile2.write(sen + '\n')
            ### get the dictionary of name entities
            all_align = []
            for key, value in alignment.items():
                all_align.append(key + '|||' + value)
            outfile3.write(' *** '.join(all_align) + '\n')
    outfile1.close()
    if args.if_anony == 'anony':  ### close files, because we do not use "with"
        outfile2.close()
        outfile3.close()
    if args.replace:
        ### replace unknown concepts with concepts in training data
        outfile4 = open(args.input_src + ".{0}.{1}".format(args.text_type, args.if_anony) + '.rep', 'w',
                        encoding="utf-8")
        train_concept_n, train_concept_a, train_concept_v, train_concept_r = get_wordnet(args.replace)
        file_to_replace = open(args.input_src + '.graph' + ".{0}".format(args.if_anony), 'r', encoding="utf-8")
        # ----------------------------------
        for line in file_to_replace:
            finish_list = []
            line_x = line.strip().split()
            line = split_sbn_list(line_x)
            for cur_sbn in line:  # the first item is WordNet
                concept_n = re.search(r'\.n\.', cur_sbn[0])
                concept_a = re.search(r'\.a\.|\.s\.', cur_sbn[0])
                concept_v = re.search(r'\.v\.', cur_sbn[0])
                concept_r = re.search(r'\.r\.', cur_sbn[0])
                if concept_n and cur_sbn[0] not in train_concept_n:  # Noun
                    new_concept = find_similar_nv(cur_sbn[0], train_concept_n)
                    print(cur_sbn[0])
                    print(new_concept)
                    cur_sbn[0] = new_concept
                if concept_v and cur_sbn[0] not in train_concept_v:  # Verb
                    new_concept = find_similar_nv(cur_sbn[0], train_concept_v)
                    print(cur_sbn[0])
                    print(new_concept)
                    cur_sbn[0] = new_concept
                if concept_a and cur_sbn[0] not in train_concept_a:  # Adjective
                    new_concept = find_similar_as(cur_sbn[0], train_concept_a)
                    print(cur_sbn[0])
                    print(new_concept)
                    cur_sbn[0] = new_concept
                if concept_r and cur_sbn[0] not in train_concept_r:  # Adverb
                    new_concept = find_similar_nv(cur_sbn[0], train_concept_r)
                    print(cur_sbn[0])
                    print(new_concept)
                    cur_sbn[0] = new_concept
                finish_list.extend([cur_sbn])
            if args.text_type == "seq":
                sbn_seq = [x for y in finish_list for x in y]
                outfile4.write(' '.join(sbn_seq) + '\n')
            else:
                sbn_graph = [' '.join(y) for y in finish_list]
                outfile4.write(' *** '.join(sbn_graph) + '\n')


