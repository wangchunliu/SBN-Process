import torch
import re
import math


def is_number(n):
    try:
        num = float(n)
        # check "nan"
        is_number = num == num  # or use `math.isnan(num)`
    except ValueError:
        is_number = False
    return is_number


def is_index(n):
    try:
        num = float(n)
        if (re.search("\+", n) or re.search("\-", n)) and not math.isnan(num):
            return True
        else:
            return False
    except ValueError:
            return False


def between_quotes(string):
    '''Return true if a value is between quotes'''
    return (string.startswith('"') and string.endswith('"')) or (string.startswith("'") and string.endswith("'"))


def include_quotes(string):
    '''Return true if a value is between quotes'''
    return string.startswith('"') or string.endswith('"') or string.startswith("'") or string.endswith("'")


def add_tuple(simple_sbn, id_cur_sbn, cur_sbn, tuple_sbn, nodes_sbn, index):
    for j in range(index, len(cur_sbn) - 2, 2):
        if is_number(cur_sbn[j + 2]) and (
                re.search("\+", cur_sbn[j + 2]) or re.search("\-", cur_sbn[j + 2])):
            try:
                nodes_sbn.append(cur_sbn[j + 1])
                tuple_sbn.append([cur_sbn[0], cur_sbn[j + 1], simple_sbn[id_cur_sbn + int(cur_sbn[j + 2])][0]])
            except:  # p04/d2291; p13/d1816
                nodes_sbn.append(cur_sbn[j + 1])
                nodes_sbn.append(cur_sbn[j + 2])
                tuple_sbn.append([cur_sbn[0], cur_sbn[j + 1], cur_sbn[j + 2]])
        else:
            nodes_sbn.append(cur_sbn[j + 1])
            nodes_sbn.append(cur_sbn[j + 2])
            tuple_sbn.append([cur_sbn[0], cur_sbn[j + 1], cur_sbn[j + 2]])
    return tuple_sbn, nodes_sbn

class SBNData:
    def __init__(self, words, traverse, graph_reent):
        self.idx = 0
        self.annotation = " ".join(words)
        self.traverse = traverse
        if len(self.traverse) == 0:
            self.parents = [-1]
        else:
            self.parents = [-1] * len(self.traverse)

        self.matrix = torch.IntTensor(3, len(self.traverse), len(self.traverse)).zero_()
        self.matrix[0, :, :] = torch.eye(len(self.traverse))
        longest_dep = 0
        for edge_reent in graph_reent:
            i, j = edge_reent
            longest_dep = max(longest_dep, j - i)
            if i == -1 or j == -1:
                continue
            if len(self.traverse) > 1:
                self.matrix[1, i, j] = 1
                self.matrix[2, j, i] = 1

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self.annotation)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.traverse)

    def __getitem__(self, key):
        return self.traverse[key]

    def __next__(self):
        self.idx += 1
        try:
            word = self.traverse[self.idx - 1]
            return word
        except IndexError:
            self.idx = 0
            raise StopIteration

    next = __next__


def parse(line):
    tuple_sbn = []
    nodes_sbn = []
    redundant_line = []  # adding Box token in line
    box_number = 0
    for cur_sbn in line:
        if len(redundant_line) == 0:
            box_number += 1
            redundant_line.append(['BOS' + str(box_number)])
        if len(cur_sbn) == 2:
            redundant_line.append(cur_sbn)
            box_number += 1
            redundant_line.append(['BOS' + str(box_number)])
        else:
            redundant_line.append(cur_sbn)
    # --------------------------------------------------------------------
    vocab_list = [] # make every token unique
    new_redundant_line = []
    for i, cur_sbn in enumerate(redundant_line):
        new_cur_sbn = cur_sbn
        for j, item in enumerate(cur_sbn):
            if is_index(item):
                continue
            elif item not in vocab_list:
                vocab_list.append(item)
            elif item in vocab_list:
                new_cur_sbn[j] = cur_sbn[j] + '^' * i+ '*' * j ## unique token
        new_redundant_line.append(new_cur_sbn)
    # --------------------------------------------------------------------
    # simplify data: No BOS and Negation # get the triples in there
    simple_line = [x for x in new_redundant_line if (x[0].isupper() == False) and ('BOS' not in x[0])] # without Negation and BOX
    box_relation = [x[0] for x in new_redundant_line if (x[0].isupper() == True) or ('BOS' in x[0])] # Negation and Box
    for i, cur_sbn in enumerate(simple_line):
        nodes_sbn.append(cur_sbn[0])
        if len(cur_sbn) == 1:
            continue
        elif ("Name" in cur_sbn[1] or "EQU" in cur_sbn[1]) and include_quotes(cur_sbn[2]):
            nodes_sbn.append(cur_sbn[1])
            if between_quotes(cur_sbn[2]) or between_quotes(cur_sbn[2].strip('*').strip('^')):
                nodes_sbn.append(cur_sbn[2])
                tuple_sbn.append([cur_sbn[0], cur_sbn[1], cur_sbn[2]])
                end_index = 2
            else:
                nodes_sbn.append(cur_sbn[2])
                tuple_sbn.append([cur_sbn[0], cur_sbn[1], cur_sbn[2]])
                for nn in range(3, len(cur_sbn), 1):
                    if cur_sbn[nn].endswith('"') or cur_sbn[nn].strip('*').strip('^').endswith('"'):
                        end_index = nn
                        for index in range(2, end_index):
                            nodes_sbn.append(cur_sbn[index+1])
                            tuple_sbn.append([cur_sbn[index], cur_sbn[index+1]])
            ### role relation after name
            if end_index + 2 < len(cur_sbn):
                tuple_sbn, nodes_sbn = add_tuple(simple_line, i, cur_sbn, tuple_sbn, nodes_sbn, index=end_index)
        else:
            tuple_sbn, nodes_sbn = add_tuple(simple_line, i, cur_sbn, tuple_sbn, nodes_sbn,index=0)
    temp_nodes_sbn = nodes_sbn + box_relation
    old_nodes_sbn = [x for y in new_redundant_line for x in y]
    ### did not change the order of BOS and dicourece reference (NEGETAIOn)
    new_nodes_sbn = [x for x in old_nodes_sbn if x in temp_nodes_sbn]
    # get bigram in tuple_sbn-------------------------------------------
    edges =[]
    for tuple in tuple_sbn:
        if len(tuple)> 2:
            edges.append([tuple[0], tuple[1]])
            edges.append([tuple[1], tuple[2]])
        else:
            edges.append([tuple[0], tuple[1]])
    sbn = new_redundant_line
    temp = ''
    for i, item in enumerate(sbn):
        for j, x in enumerate(item):
            if 'BOS' in x:
                temp = x
            elif j == 0:
                edges.append([temp, x])
    for i in range(len(sbn)-1):
        if len(sbn[i]) == 2:
            edges.append([sbn[i][0],sbn[i+1][0]])
# ---------------------------------------------------------
    ### nodes_sbn: without indices like +1, -1，have Bos and Negation; used for getting id for every words ###
    idxmap = {}
    for i, item in enumerate(new_nodes_sbn):
        idxmap[item] = i
    # ---------------------------------------------------------
    edges_novar = []
    for e in edges:
        e0 = idxmap[e[0]]
        e1 = idxmap[e[1]]
        edges_novar.append((e0, e1))
    if edges_novar == []:
        edges_novar = [(-1, 0)]
    if (-1, 0) not in edges_novar:
        edges_novar = [(-1, 0)] + edges_novar
    if (-1, 0) not in edges:
        edges = [(-1, 0)] + edges
    # ---------------------------------------------------------
    traverse = [x.strip('*').strip('^').strip('"') for x in new_nodes_sbn]
    return traverse, edges_novar, edges

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


def extract_SBN_features(line):
    global i
    if not line:
        return [], []
    words = tuple(line)
    line = split_sbn_list(line)
    traverse, graph_reent, _ = parse(line)
    return SBNData(words, traverse, graph_reent)


# ---------------------------------------------------------------------------------------
import codecs
def make_SBN_iterator_from_file(path):
    with codecs.open(path, "r", "utf-8") as corpus_file:
        for line in corpus_file:
            graph1 = extract_SBN_features(line.strip())
            print(graph1)
# make_SBN_iterator_from_file('sbn.txt')

