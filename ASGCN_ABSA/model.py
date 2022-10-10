import yaml
import math
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import initializer, HeUniform, Uniform, Normal, _calculate_fan_in_and_fan_out
from ASGCN_ABSA.utils.init import XavierNormal
#from layers.dynamic_rnn import DynamicLSTM

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

class Model(nn.Cell):
    def __init__(self, embedding_matrix):
        super(Model, self).__init__()
        with open('/code/ASGCN_ABSA/config.yaml', 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.embed = Embedding.from_pretrained_embedding(Tensor(embedding_matrix, dtype=ms.float32))
        self.text_lstm = nn.LSTM(self.cfg['embed_dim'], self.cfg['hidden_dim'], num_layers=1, batch_first=True, bidirectional=True)
        
        self.gc1 = GraphConvolution(2 * self.cfg['hidden_dim'], 2 * self.cfg['hidden_dim'])
        self.gc2 = GraphConvolution(2 * self.cfg['hidden_dim'], 2 * self.cfg['hidden_dim'])

        self.fc = Dense(2 * self.cfg['hidden_dim'], self.cfg['polarities_dim']).to_float(ms.float32)

        self.text_embed_dropout = nn.Dropout(1 - self.cfg['dropout'])

        self.relu = ops.ReLU()
        self.concat = ops.Concat(1)
        self.expand = ops.ExpandDims()
        self.squeeze = ops.Squeeze(1)
        self.softmax = ops.Softmax(2)

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.get_parameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    param.set_data(initializer(XavierNormal(), param.shape))
                else:
                    stdv = 1. / math.sqrt(param.shape[0])
                    param.set_data(initializer(Uniform(stdv), param.shape))

    def construct(self, inputs):
        text_indices = inputs[self.cfg['input_columns'][0]]
        text_len = inputs[self.cfg['input_columns'][1]]
        adj = inputs[self.cfg['input_columns'][2]]
        weight = inputs[self.cfg['input_columns'][3]]
        mask = inputs[self.cfg['input_columns'][4]]
        
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, seq_length = text_len)
        weight = self.expand(weight, 2)
        mask = self.expand(mask, 2)
        x = self.relu(self.gc1(weight * text_out, adj))
        x = self.relu(self.gc2(weight * x, adj))
        x *= mask
        alpha_mat = ops.matmul(
            x.astype(ms.float16), ops.transpose(text_out, (0, 2, 1)).astype(ms.float16)).astype(ms.float32)
        alpha = self.softmax(alpha_mat.sum(axis=1, keepdims=True))
        x = self.squeeze(ops.matmul(alpha, text_out)) 
        output = self.fc(x)
        #breakpoint()
        return output