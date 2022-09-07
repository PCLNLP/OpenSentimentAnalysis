import numpy as np
import pickle


def load_sentic_word():
    """
    load senticNet
    """
    path = '../data/senticnet_word.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = sentic
    fp.close()
    return senticNet


def dependency_adj_matrix(text, aspect, senticNet):
    word_list = text.split()
    seq_len = len(word_list)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')

    for i in range(seq_len):
        word = word_list[i]
        if word in senticNet:
            sentic = float(senticNet[word]) + 1.0
        else:
            sentic = 0
        if word in aspect:
            sentic += 1.0
        for j in range(seq_len):
            matrix[i][j] += sentic
            matrix[j][i] += sentic
    for i in range(seq_len):
        if matrix[i][i] == 0:
            matrix[i][i] = 1

    return matrix


def process(filename):
    senticNet = load_sentic_word()
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename + '.sentic', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        adj_matrix = dependency_adj_matrix(text_left + ' ' + aspect + ' ' + text_right, aspect, senticNet)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    print('done !!!', filename)
    fout.close()


if __name__ == '__main__':
    process('./data/acl14/train.raw')
    process('./data/acl14/test.raw')
    process('./data/lap14/train.raw')
    process('./data/lap14/test.raw')
    process('./data/rest14/train.raw')
    process('./data/rest14/test.raw')
    process('./data/rest15/train.raw')
    process('./data/rest15/test.raw')
    process('./data/rest16/train.raw')
    process('./data/rest16/test.raw')
