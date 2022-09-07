import math
import yaml
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.common.initializer import initializer, HeUniform, Uniform, Normal, XavierUniform, _calculate_fan_in_and_fan_out


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal', dtype=mindspore.float32,
                 padding_idx=None):
        if embedding_table == 'normal':
            embedding_table = Normal(1.0)
        super().__init__(vocab_size, embedding_size,
                         use_one_hot, embedding_table, dtype, padding_idx)

    @classmethod
    def from_pretrained_embedding(cls, embeddings: Tensor, freeze=True, padding_idx=None):
        rows, cols = embeddings.shape
        embedding = cls(rows, cols, embedding_table=embeddings,
                        padding_idx=padding_idx)
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
        self.weight = mindspore.Parameter(
            ops.zeros((in_features, out_features), mindspore.float32))
        if bias:
            self.bias = mindspore.Parameter(
                ops.zeros((out_features), mindspore.float32))
        else:
            self.bias = None

    def construct(self, text, adj):
        output = ops.matmul(text.astype(mindspore.float16), self.weight.astype(mindspore.float16)).astype(
            mindspore.float32)
        denom = adj.sum(axis=2, keepdims=True) + 1
        output = ops.matmul(adj.astype(mindspore.float16), output.astype(mindspore.float16)).astype(mindspore.float32)

        output = output / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class AAGCN(nn.Cell):
    def __init__(self, embedding_matrix):
        super(AAGCN, self).__init__()
        with open('AAGCN_ABSA/config.yaml', 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.embed = Embedding.from_pretrained_embedding(
            Tensor(embedding_matrix, dtype=mindspore.float32))
        self.text_lstm = nn.LSTM(
            self.cfg['embed_dim'], self.cfg['hidden_dim'], num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2 * self.cfg['hidden_dim'], 2 * self.cfg['hidden_dim'])
        self.gc2 = GraphConvolution(2 * self.cfg['hidden_dim'], 2 * self.cfg['hidden_dim'])
        self.gc3 = GraphConvolution(2 * self.cfg['hidden_dim'], 2 * self.cfg['hidden_dim'])
        self.gc4 = GraphConvolution(2 * self.cfg['hidden_dim'], 2 * self.cfg['hidden_dim'])

        self.fc = Dense(2 * self.cfg['hidden_dim'], self.cfg['polarities_dim']).to_float(mindspore.float16)

        self.text_embed_dropout = nn.Dropout(1 - self.cfg['dropout'])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(2)

        self.init_parameters()

    def init_parameters(self):
        for p in self.get_parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    p.set_data(initializer(XavierUniform(), p.shape, p.dtype))
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    p.set_data(initializer(Uniform(scale=stdv), p.shape, p.dtype))

    def construct(self, text_indices, adj, d_adj, seq_length):
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, _ = self.text_lstm(text, seq_length=seq_length)
        x = self.gc1(text_out, adj)
        x = self.relu(x)

        x = self.relu(self.gc2(x, d_adj))
        x = self.relu(self.gc3(x, adj))
        x = self.relu(self.gc4(x, d_adj))

        alpha_mat = ops.matmul(x.astype(mindspore.float16), text_out.swapaxes(1, 2).astype(mindspore.float16)).astype(
            mindspore.float32)
        alpha = self.softmax(alpha_mat.sum(axis=1, keepdims=True))
        x = ops.matmul(alpha.astype(mindspore.float16), text_out.astype(mindspore.float16)).astype(
            mindspore.float32).squeeze(1)

        output = self.fc(x)
        return output.astype(mindspore.float32)
