import os
import pickle
from pathlib import Path
import numpy as np
import mindspore.dataset.engine as de
from cybertron import BertTokenizer


def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines)):
                _, _, _, text_raw = lines[i].split('\t')
                text_raw = text_raw.lower().strip()
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './data/glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.load(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataSet:
    def __init__(self, data_dir, tokenize):
        self.data_dir = data_dir
        self.tokenizer = tokenize
        self.data_keys = [
            'text_bert_indices',
            'bert_segments_indices',
            'text_indices',
            'left_indices',
            'left_with_aspect_indices',
            'right_indices',
            'right_with_aspect_indices',
            'aspect_indices',
            'aspect_boundary',
            'dependency_graph',
            'sentic_graph',
            'sdat_graph',
            'polarity',
            'weight',
            'mask'
        ]
        self.dataset = self._read_data()

    @staticmethod
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

    @staticmethod
    def position_mask(left_len, aspect_len, seq_len):
        mask = []
        for j in range(left_len):
            mask.append(0)
        for j in range(left_len, min(left_len + aspect_len, seq_len)):
            mask.append(1)
        for j in range(min(left_len + aspect_len, seq_len), seq_len):
            mask.append(0)
        return mask

    def _read_data(self):
        with open(self.data_dir, 'r', encoding='utf-8', newline='\n', errors='ignore') as f1:
            with open(self.data_dir.with_suffix('.raw.graph'), 'rb') as f2:
                with open(self.data_dir.with_suffix('.raw.sentic'), 'rb') as f3:
                    with open(self.data_dir.with_suffix('.raw.graph_sdat'), 'rb') as f4:
                        lines = f1.readlines()
                        idx2graph = pickle.load(f2)
                        idx2graph_s = pickle.load(f3)
                        idx2graph_sdat = pickle.load(f4)
        all_data = []
        for i in range(0, len(lines), 3):
            graph_id = i
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            text_left = text_left.strip()
            text_right = text_right.strip()
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            context = text_left + " " + aspect + " " + text_right
            text_indices = self.tokenizer.text_to_sequence(context)
            left_indices = self.tokenizer.text_to_sequence(text_left)
            left_with_aspect_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect)
            right_indices = self.tokenizer.text_to_sequence(text_right, reverse=True)
            right_with_aspect_indices = self.tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
            aspect_indices = self.tokenizer.text_to_sequence(aspect)
            left_len = np.sum(left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)
            polarity = int(polarity) + 1

            text_len = np.sum(text_indices != 0)
            concat_bert_indices = self.tokenizer.text_to_sequence('[CLS] ' + context + ' [SEP] ' + aspect + " [SEP]")
            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)

            dependency_graph = np.pad(idx2graph[graph_id],
                                      ((0, self.tokenizer.max_seq_len - idx2graph[graph_id].shape[0]),
                                       (0, self.tokenizer.max_seq_len - idx2graph[graph_id].shape[0])), 'constant')

            sentic_graph = np.pad(idx2graph_s[graph_id],
                                  ((0, self.tokenizer.max_seq_len - idx2graph_s[graph_id].shape[0]),
                                   (0, self.tokenizer.max_seq_len - idx2graph_s[graph_id].shape[0])), 'constant')

            sdat_graph = np.pad(idx2graph_sdat[graph_id],
                                ((0, self.tokenizer.max_seq_len - idx2graph_sdat[graph_id].shape[0]),
                                 (0, self.tokenizer.max_seq_len - idx2graph_sdat[graph_id].shape[0])), 'constant')

            seq_len = self.tokenizer.max_seq_len
            weight = self.position_weight(left_len, aspect_len, text_len, seq_len)
            mask = self.position_mask(left_len, aspect_len, seq_len, )

            data = [
                concat_bert_indices,
                concat_segments_indices,
                text_indices,
                left_indices,
                left_with_aspect_indices,
                right_indices,
                right_with_aspect_indices,
                aspect_indices,
                aspect_boundary,
                dependency_graph,
                sentic_graph,
                sdat_graph,
                polarity,
                weight,
                mask
            ]
            graph_id += 1
            all_data.append(data)
        return all_data

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class ABSADataLoader:
    def __init__(self, data_dir, tokenize, batch_size=16):
        # 加载原始数据
        dataset = ABSADataSet(
            data_dir=data_dir,
            tokenize=tokenize
        )

        sequential_sampler = de.SequentialSampler()

        self.dataset = de.GeneratorDataset(
            dataset,
            dataset.data_keys,
            sampler=sequential_sampler,
            num_parallel_workers=1,
            python_multiprocessing=False
        )
        self.dataset = self.dataset.batch(batch_size, drop_remainder=False, num_parallel_workers=1)


def build_dataset(opt):
    opt.data_dir = Path(opt.data_dir)
    train_dataset_dir = opt.data_dir / opt.dataset / 'train.raw'
    test_dataset_dir = opt.data_dir / opt.dataset / 'test.raw'
    tokenize = Tokenizer4Bert(opt.max_seq_len, opt.bert_tokenizer)

    absa_train_dataset = ABSADataLoader(
        data_dir=train_dataset_dir,
        tokenize=tokenize,
        batch_size=16).dataset
    absa_test_dataset = ABSADataLoader(
        data_dir=test_dataset_dir,
        tokenize=tokenize,
        batch_size=16).dataset
    return absa_train_dataset, absa_test_dataset

