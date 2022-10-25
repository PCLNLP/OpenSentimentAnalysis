import os, math, random
from pathlib import Path
import pickle
import numpy as np
import mindspore
from mindspore import Tensor

def position_weight(left_len, aspect_len, text_len, seq_len):
    weight = []
    context_len = text_len - aspect_len
    for j in range(left_len):
        weight.append(1 - (left_len - j) / context_len)
    for j in range(left_len, min(left_len + aspect_len, seq_len)):
        weight.append(0)
    for j in range(min(left_len + aspect_len, seq_len), text_len):
        weight.append(1 - (j - (left_len + aspect_len - 1)) / context_len)
    for j in range(text_len, seq_len):
        weight.append(0)
    return weight

def position_mask(left_len, aspect_len, seq_len):
    mask = []
    for j in range(left_len):
        mask.append(0)
    for j in range(left_len, min(left_len + aspect_len, seq_len)):
        mask.append(1)
    for j in range(min(left_len + aspect_len, seq_len), seq_len):
        mask.append(0)
    return mask

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_text_len = []
        batch_context_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        batch_fre_graph = []
        batch_dependency_graph_no = []
        batch_fre_graph_no = []
        batch_post_emb = []
        batch_weight = []
        batch_mask = []
        max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            text_indices, context_indices, aspect_indices, left_indices, polarity, dependency_graph,fre_graph,dependency_graph_no,fre_graph_no,post_emb = \
                item['text_indices'], item['context_indices'], item['aspect_indices'], item['left_indices'],\
                item['polarity'], item['dependency_graph'],item['fre_graph'],item['dependency_graph_no'],item['fre_graph_no'],item['post_emb']
            text_padding = [0] * (max_len - len(text_indices))
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))
            post_padding = [0] *(max_len - len(post_emb))
            batch_text_indices.append(text_indices + text_padding)
            batch_context_indices.append(context_indices + context_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_polarity.append(polarity)
            dere_num = 5
            fere_num = 8
            redependency_graph = []
            refre_graph = []
            for i in range(dere_num):
                adj = np.pad(dependency_graph[i], \
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant')
                redependency_graph.append(adj)
            for i in range(fere_num):
                adj = np.pad(fre_graph[i], \
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant')
                refre_graph.append(adj)
            batch_dependency_graph.append(redependency_graph)
            batch_fre_graph.append(refre_graph)
            batch_dependency_graph_no.append(np.pad(dependency_graph_no,((0, max_len - len(text_indices)), (0, max_len - len(text_indices))),
                                                    'constant'))
            batch_fre_graph_no.append(np.pad(fre_graph_no, ((0, max_len - len(text_indices)), (0, max_len - len(text_indices))),
                                             'constant'))
            batch_post_emb.append(post_emb+post_padding)
            left_len = int((Tensor(left_indices) != 0).sum(-1))
            aspect_len = int((Tensor(aspect_indices) != 0).sum(-1))
            text_len = int((Tensor(text_indices) != 0).sum(-1))
            batch_text_len.append(text_len)
            #breakpoint()
            seq_len = max_len
            weight = position_weight(left_len, aspect_len, text_len, seq_len)
            mask = position_mask(left_len, aspect_len, seq_len)
            batch_weight.append(weight)
            batch_mask.append(mask)

        return { \
            'text_indices': Tensor(batch_text_indices), \
            'text_len': Tensor(batch_text_len), \
            'context_indices': Tensor(batch_context_indices), \
            'aspect_indices': Tensor(batch_aspect_indices), \
            'left_indices': Tensor(batch_left_indices), \
            'polarity': Tensor(batch_polarity), \
            'dependency_graph': Tensor(batch_dependency_graph), \
            'fre_graph': Tensor(batch_fre_graph), \
            'dependency_graph_no': Tensor(batch_dependency_graph_no), \
            'fre_graph_no': Tensor(batch_fre_graph_no), \
            'post_emb':Tensor(batch_post_emb), \
            'weight': Tensor(batch_weight), \
            'mask': Tensor(batch_mask), \
        }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]


def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32') 
            except:
                continue
    return word_vec

def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = '/data/Algorithms/data/BIGCN_ABSA_data/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix

def build_fre_matrix(feature_words_path,embed_dim,type):
    embedding_matrix_file_name = '{0}_embedding_frematrix.pkl'.format(type)
    fin = open(feature_words_path, 'rb')
    feature_words = pickle.load(fin)
    fin.close()
    if os.path.exists(embedding_matrix_file_name):
        print('loading embeddingfre_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(feature_words), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim))
        fname = '/data/Algorithms/data/BIGCN_ABSA_data/glove.840B.300d.txt'
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')

        word_vec = {}
        for line in fin:
            tokens = line.rstrip().split()
            if feature_words is None or tokens[0] in feature_words:
                try:
                    word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
                except:
                    continue
        i = 0
        for word in feature_words:
            vec = word_vec.get(word)
            if vec is not None:

                embedding_matrix[i] = vec
                i = i + 1
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))

    return embedding_matrix

class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                #每个word一个id
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words] #根据word2idx重组每句话
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer, post_vocab):
        with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            lines = f.readlines()
        with open(fname.with_suffix('.raw_hira.graph'), 'rb') as f:
            idx2gragh = pickle.load(f)
        with open(fname.with_suffix('.rawsingle_hira.graph'), 'rb') as f:
            fre_graphs = pickle.load(f)
        with open(fname.with_suffix('.raw.graph'), 'rb') as f:
            idx2gragh_no = pickle.load(f)
        with open(fname.with_suffix('.rawsingle.graph'), 'rb') as f:
            fre_graphs_no = pickle.load(f)

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            sentence = text_left+' '+aspect+' '+text_right
            sen_len = len(sentence.split())
            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_indices = tokenizer.text_to_sequence(text_left)

            aspect_len = len(aspect.split())
            left_len = len(text_left.split())
            right_len = sen_len - aspect_len - left_len

            position = list(range(-left_len,0)) + [0]*aspect_len + list(range(1,right_len + 1))
            post_emb = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in position]

            polarity = int(polarity)+1
            dependency_graph = idx2gragh[i]
            fre_graph = fre_graphs[i]

            dependency_graph_no = idx2gragh_no[i]
            fre_graph_no = fre_graphs_no[i//3]
            data = {
                'text_indices': text_indices,
                'context_indices': context_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                'fre_graph':fre_graph,
                'dependency_graph_no': dependency_graph_no,
                'fre_graph_no': fre_graph_no,
                'post_emb':post_emb

            }

            all_data.append(data)
        return all_data

    def __init__(self, opt, post_vocab=None):
        print("preparing {0} dataset ...".format(opt.dataset))
        opt.data_dir = Path(opt.data_dir)
        train_dataset_dir = opt.data_dir / opt.dataset / 'train.raw'
        test_dataset_dir = opt.data_dir / opt.dataset / 'test.raw'
        text = ABSADatesetReader.__read_text__([train_dataset_dir, test_dataset_dir])
        text = ABSADatesetReader.__read_text__([train_dataset_dir, test_dataset_dir])
        w2i_path = opt.data_dir / 'word2idx' / opt.dataset / 'word2idx.pkl'
        if w2i_path.exists():
            print("loading {0} tokenizer...".format(opt.dataset))
            with open(w2i_path, 'rb') as f:
                 word2idx = pickle.load(f)
                 tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(w2i_path, 'wb') as f:
                 pickle.dump(tokenizer.word2idx, f)
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim=300, type = opt.dataset)
        fre_embedding = self.embedding_matrix
        self.fre_embedding = mindspore.Tensor(fre_embedding, mindspore.float32)
        fin = open(train_dataset_dir.with_suffix('.rawfre_full_all.graph'), 'rb')
        common_adj = pickle.load(fin)
        len_1 = len(tokenizer.word2idx)

        row = len(common_adj[0])
        diff = len_1 - row
        fin.close()
        cp_row = np.zeros((diff, row))
        common_adj = np.insert(common_adj,0,values=cp_row,axis=0)

        cp_col = np.zeros((diff,len_1))
        common_adj = np.insert(common_adj,0,values=cp_col,axis=1)

        self.common_adj = mindspore.Tensor(common_adj, mindspore.float32)


        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(train_dataset_dir, tokenizer, post_vocab))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(test_dataset_dir, tokenizer, post_vocab))