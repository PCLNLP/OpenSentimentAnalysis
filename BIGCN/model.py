from inspect import Parameter
from numpy import mask_indices
import yaml
import math
import mindspore as ms
from mindspore import nn, Tensor, ops
from mindspore.common.initializer import initializer, HeUniform, Uniform, Normal, _calculate_fan_in_and_fan_out
from BIGCN_ABSA.utils.init import XavierNormal

EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

class DynamicLSTM(nn.Cell):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0.0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type = 'LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        
        if self.rnn_type == 'LSTM': 
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                has_bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)  
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                has_bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                has_bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        

    def construct(self, x, x_len, h0=None):

        _, x_sort_idx = ops.Sort()(-x_len.astype("float32"))
        _, x_unsort_idx = ops.Sort()(x_sort_idx.astype("float32"))

        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        
        if self.rnn_type == 'LSTM':
            if h0 is None: 
                out, (ht, ct) = self.RNN(x, None)
            else:
                out, (ht, ct) = self.RNN(x, (h0, h0))
        else: 
            if h0 is None:
                out, ht = self.RNN(x, None)
            else:
                out, ht = self.RNN(x, h0)
            ct = None
        """unsort: h"""
        ht = ops.transpose(ht, (1, 0, 2))[x_unsort_idx]
        ht = ops.transpose(ht, (1, 0, 2))

        if self.only_use_last_hidden_state:
            return ht
        else:

            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type =='LSTM':
                ct = ops.transpose(ct, (1, 0, 2))[x_unsort_idx]
                ct = ops.transpose(ct, (1, 0, 2))
            return out, (ht, ct)

class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal', dtype=ms.float32,
                 padding_idx=None):
        if embedding_table == 'normal':
            embedding_table = Normal(1.0)
        super().__init__(vocab_size, embedding_size, use_one_hot, embedding_table, dtype, padding_idx)

    @classmethod
    def from_pretrained_embedding(cls, embeddings: Tensor, freeze=True, padding_idx=None):
        rows, cols = embeddings.shape
        embedding = cls(rows, cols, embedding_table=embeddings, padding_idx=padding_idx)
        embedding.embedding_table.requires_grad = not freeze

        return embedding


class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None):
        super().__init__(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=has_bias,
                         activation=activation)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))


class GraphConvolution(nn.Cell):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = ms.Parameter(
            ops.ones((in_features, out_features), ms.float32))
        if bias:
            self.bias = ms.Parameter(
                ops.ones(out_features, ms.float32))
        else:
            self.bias = None

    def construct(self, text, adj):
        hidden = ops.matmul(text.astype(ms.float16), self.weight.astype(ms.float16)).astype(ms.float32)
        denom = adj.sum(axis=2, keepdims=True) + 1
        output = ops.matmul(adj.astype(ms.float16), hidden.astype(ms.float16)).astype(ms.float32) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphConvolutionFRE(nn.Cell):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionFRE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = ms.Parameter(
            ops.ones((in_features, out_features), ms.float32))
        if bias:
            self.bias = ms.Parameter(
                ops.ones(out_features, ms.float32))
        else:
            self.bias = None

    def construct(self, text, adj):
        hidden = ops.matmul(text.astype(ms.float16), self.weight.astype(ms.float16)).astype(ms.float32)
        deom = adj > 0.5
        #deom = ms.Parameter(deom.astype(ms.float32), requires_grad=False)
        denom = deom.sum(axis=1, keepdims=True) + 1
        output = ops.matmul(adj.astype(ms.float16), hidden.astype(ms.float16)).astype(ms.float32) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphConvolutionRE(nn.Cell):
    def __init__(self, in_features, out_features, rela_len, bias=True, frc_lin=False):
        super(GraphConvolutionRE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rela_len = rela_len
        self.weight = ms.Parameter(
            ops.ones((in_features, out_features//rela_len), ms.float32))
        self.frc_lin = frc_lin
        if self.frc_lin == True:
           self.fre_line = Dense((out_features//rela_len)*rela_len, out_features)
        if bias:
            self.bias = ms.Parameter(
                ops.ones(out_features, ms.float32))
        else:
            self.bias = None

    def construct(self, text, adj_re):
        hidden = ops.matmul(text.astype(ms.float16), self.weight.astype(ms.float16)).astype(ms.float32)
        adj_re = ops.transpose(adj_re, (1,0,2,3))
        denom1 = adj_re[0].sum(axis=2, keepdims=True) + 1
        output = ops.matmul(adj_re[0].astype(ms.float16), hidden.astype(ms.float16)).astype(ms.float32) / denom1
        output = ops.Softmax(-1)(output)
        
        for i in range(1,self.rela_len):
            denom2 = adj_re[i].sum(axis=2, keepdims=True) + 1
            output2 = ops.matmul(adj_re[i].astype(ms.float16), hidden.astype(ms.float16)).astype(ms.float32) / denom2
            output2 = ops.ReLU()(output2)
            output = ops.Concat(2)((output, output2))

        if self.frc_lin:
            output = self.fre_line(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Model(nn.Cell):
    def __init__(self, embedding_matrix, common_adj, fre_embedding, post_vocab):
        super(Model, self).__init__()
        with open('./BIGCN_ABSA/config.yaml', 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        D = self.cfg['embed_dim']
        Co = self.cfg['kernel_num']
        self.num_layer = self.cfg['num_layer']
        self.post_vocab = post_vocab
        self.post_size = len(post_vocab)
        self.common_adj = common_adj
        self.fre_embedding = fre_embedding
        self.embed = Embedding.from_pretrained_embedding(Tensor(embedding_matrix, dtype=ms.float32))
        self.post_embed = Embedding(self.post_size, self.cfg['post_dim'], padding_idx=PAD_ID) if self.cfg['post_dim'] > 0 else None
        self.text_lstm = nn.LSTM(self.cfg['hidden_dim']+self.cfg['post_dim'], self.cfg['hidden_dim']//2, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolutionRE(self.cfg['hidden_dim'], self.cfg['hidden_dim'], 5, frc_lin=True)
        self.gc2 = GraphConvolutionRE(self.cfg['hidden_dim'], self.cfg['hidden_dim'], 5, frc_lin=True)
        self.gc3 = GraphConvolutionRE(self.cfg['hidden_dim'], self.cfg['hidden_dim'], 8, frc_lin=True)
        self.gc4 = GraphConvolutionRE(self.cfg['hidden_dim'], self.cfg['hidden_dim'], 8, frc_lin=True)
        self.gc5 = GraphConvolutionFRE(self.cfg['hidden_dim'], self.cfg['hidden_dim'])
        self.gc6 = GraphConvolutionFRE(self.cfg['hidden_dim'], self.cfg['hidden_dim'])
        self.fc = Dense(2*self.cfg['hidden_dim'], self.cfg['polarities_dim'])
        self.text_embed_dropout = nn.Dropout(1 - self.cfg['dropout'])
        self.convs3 = nn.CellList([nn.Conv1d(D, Co, K, pad_mode='pad', padding=K - 2) for K in [3]])
        self.fc_aspect = Dense(128, 2*D)
        self.att_line = Dense(self.cfg['hidden_dim'],2*self.cfg['hidden_dim'])

        self.weight = ms.Parameter(ops.ones((2 * self.cfg['hidden_dim'], 2 * self.cfg['hidden_dim']), ms.float32))
        self.bias = ms.Parameter(ops.ones((2 * self.cfg['hidden_dim']), ms.float32))

        self.relu = ops.ReLU()
        self.softmax = ops.Softmax(-1)
        self.concat1 = ops.Concat(1)
        self.concat2 = ops.Concat(2)
        self.transpose = ops.Transpose()
        self.squeeze1 = ops.Squeeze(1)
        self.squeeze2 = ops.Squeeze(2)
        self.expand = ops.ExpandDims()
        self.tanh = ops.Tanh()
        #self.maxpool1d = nn.MaxPool1d()

        self.text_fre = Embedding.from_pretrained_embedding(self.pre_frequency())

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.get_parameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    param.set_data(initializer(XavierNormal(), param.shape))
                else:
                    stdv = 1. / math.sqrt(param.shape[0])
                    param.set_data(initializer(Uniform(stdv), param.shape))

    def pre_frequency(self):
        em = self.relu(self.softmax(self.gc5(self.fre_embedding, self.common_adj)))
        pre_em = self.relu(self.softmax(self.gc6(em, self.common_adj)))
        return pre_em

    def cross_network(self,f0,fn):
        fn_weight = ops.matmul(fn,self.weight)
        fl = f0*fn_weight + self.bias + f0
        x = fl[:,:,0:self.cfg['hidden_dim']]
        y = fl[:,:,self.cfg['hidden_dim']:]
        return x,y

    def construct(self, inputs):
        text_indices = inputs[self.cfg['input_columns'][0]]
        text_len = inputs[self.cfg['input_columns'][1]]
        aspect_indices = inputs[self.cfg['input_columns'][2]]
        adj = inputs[self.cfg['input_columns'][3]]
        adj2 = inputs[self.cfg['input_columns'][4]]
        post_emb = inputs[self.cfg['input_columns'][5]]
        weight = inputs[self.cfg['input_columns'][6]]
        mask = inputs[self.cfg['input_columns'][7]]

        text = [self.embed(text_indices)]
        if self.cfg['post_dim'] > 0:
            text +=[self.post_embed(post_emb)]
        text = self.concat2(text)
        text = self.text_embed_dropout(text)
        #breakpoint()
        text_out, (_, _) = self.text_lstm(text, seq_length = text_len)
        
        weight = self.expand(weight, 2)
        mask = self.expand(mask, 2)

        #text_fre_embedding = self.pre_frequency()
        #text_fre = Embedding.from_pretrained_embedding(text_fre_embedding)
        text_out_fre = self.text_fre(text_indices)
        text_out_fre = self.text_embed_dropout(text_out_fre)

        f0 = self.concat2([text_out_fre,text_out]) #x:fre  y:syn
        numlayer = self.num_layer
        f_n = f0
        for i in range(numlayer):
            if i == 0:
                x, y = self.cross_network(f0,f0)
                x = self.softmax(self.gc3(weight * x, adj2))
                y = self.softmax(self.gc1(weight * y, adj))
                f_n = self.concat2([x,y])
            else:#多层的更新
                x,y = self.cross_network(f0,f_n)
                x = self.softmax(self.gc4(weight * x, adj2))
                y = self.softmax(self.gc2(weight * y, adj))
                f_n = self.concat2([x,y])

        aspect = self.embed(aspect_indices)
        
        aa = []
        for conv in self.convs3:
            aa.append(self.relu(conv(self.transpose(aspect, (0, 2 ,1)))))
        #aa = [self.relu(conv(self.transpose(aspect, (0, 2 ,1)))) for conv in self.convs3]
        temp = []
        for a in aa:
            temp.append(self.squeeze2(nn.MaxPool1d(a.shape[2])(a)))
        #aa = [self.squeeze2(nn.MaxPool1d(a.shape[2])(a)) for a in aa]
        aa = temp
        aspect_v = self.concat1(aa)
        temp = self.expand(self.fc_aspect(aspect_v), 2)
        aa2 = self.tanh(f_n + self.transpose(temp, (0, 2, 1)))
        xaa = f_n * aa2
        xaa2 = self.transpose(xaa, (0, 2, 1))
        xaa2 = self.squeeze2(nn.MaxPool1d(xaa2.shape[2])(xaa2))

        f_n_mask = f_n * mask

        text_out = self.att_line(text_out)
        alpha_mat = ops.matmul(f_n_mask, self.transpose(text_out, (0, 2, 1)))
        alpha = self.softmax(alpha_mat.sum(axis=1, keepdims=True))
        x = self.squeeze1(ops.matmul(alpha, text_out))
        x = self.relu(x + xaa2)
        output = self.fc(x)
        #breakpoint()
        return output