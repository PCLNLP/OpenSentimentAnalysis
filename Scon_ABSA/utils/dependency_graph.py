# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle

from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].strip()
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)        
    fout.close() 

def cl2X3_process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in range(0, len(lines), 5):
        text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].strip()
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()


if __name__ == '__main__':

    process("./data/cl2X3_acl2014/test.raw")
    process("./data/cl2X3_lap2014/test.raw")
    process("./data/cl2X3_res2014/test.raw")
    process("./data/cl2X3_res2015/test.raw")
    process("./data/cl2X3_res2016/test.raw")
    process("./data/cl2X3_mams/test.raw")
    
    cl2X3_process("./data/cl2X3_acl2014/train.raw")
    cl2X3_process("./data/cl2X3_lap2014/train.raw")
    cl2X3_process("./data/cl2X3_res2014/train.raw")
    cl2X3_process("./data/cl2X3_res2015/train.raw")
    cl2X3_process("./data/cl2X3_res2016/train.raw")
    cl2X3_process("./data/cl2X3_mams/train.raw")


    pass