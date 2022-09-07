import numpy as np
import spacy
import pickle

nlp = spacy.load('en_core_web_sm')


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
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    # print('='*20+':')
    # print(document)
    # print(senticNet)

    for token in document:
        # print('token:', token)
        if str(token) in senticNet:
            sentic = float(senticNet[str(token)]) + 1
        else:
            sentic = 0
        if str(token) in aspect:
            sentic += 1
        if token.i < seq_len:
            matrix[token.i][token.i] = 1 * sentic
            # https://spacy.io/docs/api/token
            for child in token.children:
                if str(child) in aspect:
                    sentic += 1
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1 * sentic
                    matrix[child.i][token.i] = 1 * sentic

    return matrix


def process(filename):
    senticNet = load_sentic_word()
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename + '.graph_sdat', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        adj_matrix = dependency_adj_matrix(text_left + ' ' + aspect + ' ' + text_right, aspect, senticNet)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    print('done !!!' + filename)
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
