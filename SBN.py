import torch
import re

def is_number(n):
    try:
        num = float(n)
        # 检查 "nan"
        is_number = num == num  # 或者使用 `math.isnan(num)`
    except ValueError:
        is_number = False
    return is_number

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
    def __init__(self, words, traverse, graph_edge, graph_noedge, graph_seq):
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
        for edge_reent in graph_edge:
            i, j = edge_reent
            longest_dep = max(longest_dep, j - i)
            if i == -1 or j == -1:
                continue
            if len(self.traverse) > 1:
                self.matrix[1, i, j] = 1
                self.matrix[2, j, i] = 1
        # for edge_no_reent in graph_noedge:
        #     i, j = edge_no_reent
        #     longest_dep = max(longest_dep, j - i)
        #     if i == -1 or j == -1:
        #         continue
        #     if len(self.traverse) > 1:
        #         self.matrix[3, i, j] = 1
        #         self.matrix[4, j, i] = 1
        #for edge_seq in graph_seq:
        #    i, j = edge_seq
        #    longest_dep = max(longest_dep, j - i)
        #    if i == -1 or j == -1:
        #        continue
        #    if len(self.traverse) > 1:
        #        self.matrix[3, i, j] = 1
        #        self.matrix[4, j, i] = 1

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
    edges = []
    edges_novar = []  ## no edge nodes
    edges_seqvar = []  ## only for seqence edge
    # -------------------------------------------------------------------
    redundant_line = []  # 冗余的序列，增加了Box信息
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
    vocab_list = []  # 如果列表中的元素重复则更新，使其每个元素都独一无二
    new_redundant_line = []
    for i, cur_sbn in enumerate(redundant_line):
        new_cur_sbn = cur_sbn
        for j, item in enumerate(cur_sbn):
            if is_number(item) and (re.search("\+", item) or re.search("\-", item)):
                continue
            elif item not in vocab_list:
                vocab_list.append(item)
            elif item in vocab_list:
                new_cur_sbn[j] = cur_sbn[j] + '^' * i + '*' * j
        new_redundant_line.append(new_cur_sbn)
    # --------------------------------------------------------------------
    # 简化的数据: 没有BOS 和 Negation 信息 # 得到所有的三元组
    simple_line = [x for x in new_redundant_line if (x[0].isupper() == False)] # 简化的序列，没有Negation 信息
    box_relation = [x[0] for x in new_redundant_line if (x[0].isupper() == True) or ('BOS' in x[0])]
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
            ###超过name以后的role relation
            if end_index + 2 < len(cur_sbn):
                tuple_sbn, nodes_sbn = add_tuple(simple_line, i, cur_sbn, tuple_sbn, nodes_sbn, index=end_index)
        else:
            tuple_sbn, nodes_sbn = add_tuple(simple_line, i, cur_sbn, tuple_sbn, nodes_sbn,index=0)
            # --------------------------------------------------------------------
    temp_nodes_sbn = nodes_sbn + box_relation
    old_nodes_sbn = [x for y in new_redundant_line for x in y]
    new_nodes_sbn = [x for x in old_nodes_sbn if x in temp_nodes_sbn]
    # 得到tuple_sbn中的二元组
    for tuple in tuple_sbn:
        if len(tuple)>2:
            edges.append([tuple[0], tuple[1]])
            edges.append([tuple[1], tuple[2]])
            edges_novar.append([tuple[0], tuple[2]])
        else:
            edges.append([tuple[0], tuple[1]])
            edges_novar.append([tuple[0], tuple[1]])
    ### add sequence order edge
    for i, cur_sbn in enumerate(line):
        if i > 0:
            edges_seqvar.append([line[i-1][0], line[i][0]])
        else:
            continue
    ### add BOS and NEAGTION edges
    sbn = new_redundant_line 
    temp = ''
    for i, item in enumerate(sbn):
        for j, x in enumerate(item):
            if 'BOS' in x:
                temp = x
            elif j == 0:
                edges.append([temp, x])
    for i in range(len(sbn) - 1):
        if len(sbn[i]) == 2:
            edges.append([sbn[i][0], sbn[i + 1][0]])
    # ---------------------------------------------------------
    ### nodes_sbn: 没有 +1 和 -1 这些指向信息，但是有Bos和Negation信息，用于得到数据中每个word的id ###
    idxmap = {}
    for i, item in enumerate(new_nodes_sbn):
        idxmap[item] = i
    # ---------------------------------------------------------
    edges_all = []
    for e in edges:
        e0 = idxmap[e[0]]
        e1 = idxmap[e[1]]
        edges_all.append((e0, e1))
    if edges_all == []:
        edges_all = [(-1, 0)]
    if (-1, 0) not in edges_all:
        edges_all = [(-1, 0)] + edges_all
    # ---------------------------------------------------------
    edges_noedge = []
    for e in edges_novar:
        e0 = idxmap[e[0]]
        e1 = idxmap[e[1]]
        edges_noedge.append((e0, e1))
    if edges_noedge == []:
        edges_noedge = [(-1, 0)]
    if (-1, 0) not in edges_noedge:
        edges_noedge = [(-1, 0)] + edges_noedge
    # ---------------------------------------------------------
    edges_seq = []
    for e in edges_seqvar:
        e0 = idxmap[e[0]]
        e1 = idxmap[e[1]]
        edges_seq.append((e0, e1))
    if edges_seq == []:
        edges_seq = [(-1, 0)]
    if (-1, 0) not in edges_seq:
        edges_seq = [(-1, 0)] + edges_seq
    # ---------------------------------------------------------
    traverse = [x.strip('^').strip('"') for x in new_nodes_sbn]
    return traverse, edges_all, edges_noedge, edges_seq

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
    # line = line.strip().split()
    line = split_sbn_list(line)
    traverse, graph_edge, graph_noedge, graph_seq = parse(line)
    return SBNData(words, traverse, graph_edge, graph_noedge, graph_seq)

# ---------------------------------------------------------------------------------------
line1 = [['NEGATION', '-1'], ['cut.v.01', 'Time', '+1', 'Agent', '+2', 'Patient', '+3'], ['time.n.08', 'EQU', 'now'], ['person.n.01'], ['ginger.n.02']]
line2 = [['general_election.n.01'], ['country.n.02', 'Name', '"Denmark"'], ['male.n.02', 'Name', '"Poul', 'Schlüter"'], ['coalition.n.01'],['time.n.08', 'TPR', 'now'], ['party.n.01', 'Name', '"Danish', 'Social', 'Liberal', 'Party"'], ['leave.v.01', 'Theme', '-1']]
line4 = ['male.n.02', 'Name', 'NAMEname0', '***', 'leave.v.01', 'Theme', '-1', 'Time', '+1', 'Destination', '+2', '***', 'time.n.08', 'TPR', 'now', 'TIN', '+3', '***', 'city.n.01', 'Name', '"Boston"', '***', 'day.n.03', '***', 'day.n.03', 'TAB', '-1', '***', 'terra_incognita.n.01', 'EQU', 'now', 'TIN', '-2']
line5 = ['NEGATION', '-1', '***','person.n.01', '***', 'name.v.01', 'Agent', '-1', 'Time', '+1', 'Theme', '+3', 'Result', '+4', '***', 'time.n.08', 'TPR', 'now', '***', 'baby.n.01', '***', 'name.n.01', 'EQU', '"Jane', 'Johan"']
line6 = ['NEGATION', '-1', '***', 'male.n.02', '***', 'time.n.08', 'EQU', 'now', '***', 'play.v.01', 'Agent', '-2', 'Time', '-1', 'Theme', '+1', '***', 'monopoly.n.03']

import codecs
def make_SBN_iterator_from_file(path):
    with codecs.open(path, "r", "utf-8") as corpus_file:
        for line in corpus_file:
            graph1 = extract_SBN_features(line.strip())
            print(graph1.matrix)
            print(graph1.traverse)

# make_SBN_iterator_from_file('sbn.txt')

# graph1 = extract_SBN_features(line5)
# print(graph1.matrix)
# print(graph1.traverse)
# print(graph1.parents)
# print(graph1.annotation)
#
