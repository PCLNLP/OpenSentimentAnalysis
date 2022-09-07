# -*- coding: utf-8 -*-
# ------------------
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

import numpy as np
import spacy
import pickle


def dependency_adj_matrix(text, aspect, position, other_aspects):
    text_list = text.split()
    seq_len = len(text_list)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    position = int(position)
    flag = 1

    for i in range(seq_len):
        word = text_list[i]
        if word in aspect:
            for other_a in other_aspects:
                other_p = int(other_aspects[other_a])
                add = 0
                for other_w in other_a.split():
                    weight = 1 + (1 / (abs(add + other_p - position) + 1))
                    matrix[i][other_p + add] = weight
                    matrix[other_p + add][i] = weight
                    add += 1
    return matrix


def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename + '.graph_af', 'wb')
    graph_idx = 0
    for i in range(len(lines)):
        aspects, polarities, positions, text = lines[i].split('\t')
        aspect_list = aspects.split('||')
        polarity_list = polarities.split('||')
        position_list = positions.split('||')
        text = text.lower().strip()
        aspect_graphs = {}
        aspect_positions = {}
        for aspect, position in zip(aspect_list, position_list):
            aspect_positions[aspect] = position
        for aspect, position in zip(aspect_list, position_list):
            aspect = aspect.lower().strip()
            other_aspects = aspect_positions.copy()
            del other_aspects[aspect]
            adj_matrix = dependency_adj_matrix(text, aspect, position, other_aspects)
            idx2graph[graph_idx] = adj_matrix
            graph_idx += 1
    pickle.dump(idx2graph, fout)
    print('done !!!' + filename)
    fout.close()


if __name__ == '__main__':
    process('./data/rest14/train.raw')
    process('./data/rest14/test.raw')
    process('./data/lap14/train.raw')
    process('./data/lap14/test.raw')
    process('./data/rest15/train.raw')
    process('./data/rest15/test.raw')
    process('./data/rest16/train.raw')
    process('./data/rest16/test.raw')
