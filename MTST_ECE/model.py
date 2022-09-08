import math
import yaml
import mindspore
from mindspore import Tensor
from mindspore.common.initializer import initializer, HeUniform, Uniform, Normal, _calculate_fan_in_and_fan_out
from mindspore import nn
from mindspore import ops
from mindspore import dtype as mstype
from cybertron import BertModel

from MTST_ECE.utils.init import XavierNormal


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


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_size):
        super().__init__(vocab_size, embedding_size, use_one_hot=False, embedding_table='normal',
                         dtype=mstype.float32, padding_idx=None)
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding_table.set_data(initializer(Normal(sigma=1.0, mean=0.0), self.embedding_table.shape))


class BertEncoder(nn.Cell):
    def __init__(self, pretrained_bert_name):
        super().__init__()
        self.bert = BertModel.load(pretrained_bert_name)

    def construct(self, ids_padding_tensor, mask_tensor, document_len):
        _, pooled = self.bert(ids_padding_tensor, attention_mask=mask_tensor)
        start = 0
        clause_state_list = []
        for dl in document_len:
            end = start + dl
            clause_state_list.append(pooled[start: end])
            start = end
        return pooled, clause_state_list


class SLModel(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.is_bi = config['is_bi']
        self.bert_output_size = config['bert_output_size']
        self.mlp_size = config['mlp_size']
        self.cell_size = config['cell_size']
        self.scale_factor = config['scale_factor']
        self.dropout = config['dropout']
        self.layers = config['layers']
        self.gamma = config['gamma']
        self.scope = config['scope']
        self.tag_ebd_dim = config['tag_ebd_dim']
        self.tag_embedding = Embedding(self.scope * 2 + 1 + 1 + 1 + 1, self.tag_ebd_dim)

        self.encoder = nn.LSTM(self.bert_output_size, self.cell_size, self.layers, bidirectional=self.is_bi)
        self.decoder = nn.LSTM(
            self.cell_size * 2 + self.tag_ebd_dim if self.is_bi else self.cell_size + self.tag_ebd_dim,
            self.cell_size, self.layers, bidirectional=False)
        self.tag_MLP = nn.SequentialCell(
            Dense(self.cell_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(alpha=0.01),
            nn.Dropout(1 - self.dropout),
            Dense(self.mlp_size, self.scope * 2 + 1 + 1)
        )
        self.emo_MLP = nn.SequentialCell(
            Dense(self.cell_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(alpha=0.01),
            nn.Dropout(1 - self.dropout),
            Dense(self.mlp_size, 2)
        )
        self.cau_MLP = nn.SequentialCell(
            Dense(self.cell_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(alpha=0.01),
            nn.Dropout(1 - self.dropout),
            Dense(self.mlp_size, 2)
        )

        self.init_weight()

    def init_weight(self):
        for param in self.get_parameters():
            if param.name.find("weight") != -1 or param.name.find("embedding_table") != -1:
                if len(param.shape) > 1:
                    param.set_data(initializer(XavierNormal(), param.shape))
                else:
                    param.set_data(initializer(Uniform(0.1), param.shape))
            elif param.name.find("bias") != -1:
                param.set_data(initializer(Uniform(0.1), param.shape))
            else:
                continue
        self.tag_embedding.embedding_table.requires_grad = False

    def init_hidden(self, batch_size, mode):

        if self.is_bi and mode == 'encoder':
            hidden = (Tensor(initializer('zeros', (self.layers * 2, batch_size, self.cell_size))),
                      Tensor(initializer('zeros', (self.layers * 2, batch_size, self.cell_size))))
        else:
            hidden = (Tensor(initializer('zeros', (self.layers, batch_size, self.cell_size))),
                      Tensor(initializer('zeros', (self.layers, batch_size, self.cell_size))))
        return hidden

    def Encoder(self, clause_state_list, len_list):
        state_dim = clause_state_list[0].shape[1]
        bs = len(clause_state_list)
        max_len = max(len_list)
        clause_padding_list = []
        for x in clause_state_list:
            len_gap = max_len - x.shape[0]
            clause_padding = x
            if len_gap != 0:
                clause_padding = ops.Concat()([clause_padding, Tensor(initializer('zeros', (len_gap, state_dim)))])
            clause_padding_list.append(clause_padding)
        en_inputs = ops.transpose(ops.Stack()(clause_padding_list), (1, 0, 2))
        init_hidden = self.init_hidden(bs, 'encoder')
        en_outputs, _ = self.encoder(en_inputs, init_hidden)
        en_outputs = ops.transpose(en_outputs, (1, 0, 2))

        return en_outputs

    def Decoder(self, en_outputs, len_list, tag_labels):
        start, bs = 0, len(len_list)
        tag_inputs = []
        max_len = max(len_list)
        for dl in len_list:
            end = start + dl
            tags = tag_labels[start: end]
            temp = [self.scope * 2 + 1 + 1] + tags[:-1] + [self.scope * 2 + 1 + 1 + 1] * (max_len - dl)
            tag_inputs.append(Tensor(temp, mindspore.int64))
            start = end
        tag_inputs_tensor = ops.Stack()(tag_inputs)
        tag_inputs_ebd = self.tag_embedding(tag_inputs_tensor)
        de_inputs = ops.transpose(ops.Concat(2)([en_outputs, tag_inputs_ebd]), (1, 0, 2))
        init_state = self.init_hidden(bs, 'decoder')
        de_outputs, _ = self.decoder(de_inputs, init_state)
        de_outputs = ops.transpose(de_outputs, (1, 0, 2))

        return de_outputs

    def ProbsBias(self, i, tag_probs_seg, cau_probs_seg, e_i):

        num_seg = tag_probs_seg.shape[0]
        num_pre = self.scope - i if self.scope - i > 0 else 0
        top_padding = ops.Zeros()((num_pre, self.scope * 2 + 1 + 1), mindspore.float32)
        cau_probs_seg = ops.Concat()([Tensor([0] * num_pre, mindspore.float32), cau_probs_seg])
        num_tail = self.scope - (num_seg - i - 1) if self.scope - (num_seg - i - 1) > 0 else 0
        tail_padding = ops.Zeros()((num_tail, self.scope * 2 + 1 + 1), mindspore.float32)
        cau_probs_seg = ops.Concat()([cau_probs_seg, Tensor([0] * num_tail, mindspore.float32)])
        padding_seg = ops.Concat()([top_padding, tag_probs_seg, tail_padding])

        bias = []
        for i in range(self.scope * 2 + 1):
            total = 1 - padding_seg[i][self.scope * 2 - i]
            p_weight = 1 - (abs(i - self.scope) + self.gamma) / (self.scope + self.gamma * 2)
            e_weight = e_i
            c_weight = cau_probs_seg[i]
            if e_i > 0.5:
                v = c_weight * e_weight * p_weight * total
            else:
                v = (1 - c_weight) * (1 - e_weight) * (1 - p_weight) * (1 - total)
            temp = [-v / (self.scope * 2 + 1) if _ != self.scope * 2 - i else v for _ in range(self.scope * 2 + 1 + 1)]
            temp = ops.Stack()(temp)
            bias.append(Tensor(temp))
        bias = bias[num_pre: num_pre + num_seg]
        bias = ops.Stack(0)(bias)
        return bias

    def ProbsRevised(self, tag_probs_i, emo_i, cau_i):
        for i, e_i in enumerate(emo_i):
            len_i = tag_probs_i.shape[0]
            start = 0 if i - self.scope < 0 else i - self.scope
            end = -1 if i + self.scope > len_i - 1 else i + self.scope + 1
            tag_probs_seg = tag_probs_i[start: end]
            cau_probs_seg = cau_i[start: end]
            bias = self.ProbsBias(i, tag_probs_seg, cau_probs_seg, e_i)

            if e_i < 0.5:
                tag_probs_i[start: end] = tag_probs_i[start: end] - bias
            else:
                tag_probs_i[start: end] = tag_probs_i[start: end] + bias

        return tag_probs_i

    def ProbsDrifter(self, tag_probs, emo_probs, cau_probs, len_list):
        retag_probs_list = []
        start = 0
        for l in len_list:
            end = start + l
            tag_probs_i = tag_probs[start: end]
            emo_probs_i = emo_probs[start: end]
            cau_probs_i = cau_probs[start: end]
            emo_i, cau_i = emo_probs_i[:, 1], cau_probs_i[:, 1]
            retag_probs_i = self.ProbsRevised(tag_probs_i, emo_i, cau_i)
            start = end
            retag_probs_list.append(retag_probs_i)

        retag_probs = ops.Concat()(retag_probs_list)
        return retag_probs

    def Eval(self, en_outputs, len_list):
        bs = len(len_list)
        inputs = ops.transpose(en_outputs, (1, 0, 2))
        tag_inputs = [self.scope * 2 + 1 + 1] * bs
        tag_inputs_tensor = Tensor(tag_inputs, mindspore.int64)
        tags_inputs_ebd = ops.ExpandDims()(self.tag_embedding(tag_inputs_tensor), 0)
        init_state = self.init_hidden(bs, 'decoder')
        feature_list = []
        for step_i in inputs:
            step_i = ops.ExpandDims()(step_i, 0)
            inputs_i = ops.Concat(2)([step_i, tags_inputs_ebd])
            output, init_state = self.decoder(inputs_i, init_state)
            feature_list.append(ops.Squeeze(0)(output))
            tag_inputs = self.tag_MLP(ops.Squeeze(0)(output)).argmax(1).asnumpy().tolist()
            tag_inputs_tensor = Tensor(tag_inputs, mindspore.int64)
            tags_inputs_ebd = ops.ExpandDims()(self.tag_embedding(tag_inputs_tensor), 0)
        features = ops.transpose(ops.Stack()(feature_list), (1, 0, 2))
        features = ops.Concat()([features[i][:l] for i, l in enumerate(len_list)])
        tag_logits = self.tag_MLP(features)
        emo_logits = self.emo_MLP(features)
        cau_logits = self.cau_MLP(features)

        tag_probs = ops.Softmax(1)(tag_logits)
        emo_probs = ops.Softmax(1)(emo_logits)
        cau_probs = ops.Softmax(1)(cau_logits)
        retag_probs = self.ProbsDrifter(tag_probs, emo_probs, cau_probs, len_list)

        return retag_probs, emo_probs, cau_probs

    def construct(self, clause_state_list, tag_labels, mode):
        len_list = [x.shape[0] for x in clause_state_list]
        if mode == 1:
            en_outputs = self.Encoder(clause_state_list, len_list)
            de_outputs = self.Decoder(en_outputs, len_list, tag_labels)
            de_outputs_list = []
            for i, l in enumerate(len_list):
                de_outputs_list.append(de_outputs[i][:l])

            features = ops.Concat()(de_outputs_list)
            tag_logits = self.tag_MLP(features)
            emo_logits = self.emo_MLP(features)
            cau_logits = self.cau_MLP(features)

            tag_probs = ops.Softmax(1)(tag_logits)
            emo_probs = ops.Softmax(1)(emo_logits)
            cau_probs = ops.Softmax(1)(cau_logits)

            emo_log_probs = ops.Log()(emo_probs)
            cau_log_probs = ops.Log()(cau_probs)
            tag_log_probs = ops.Log()(tag_probs)
            return tag_log_probs, emo_log_probs, cau_log_probs
        else:
            en_outputs = self.Encoder(clause_state_list, len_list)
            retag_probs, emo_probs, cau_probs = self.Eval(en_outputs, len_list)

            return retag_probs, emo_probs, cau_probs


class Model(nn.Cell):

    def __init__(self):
        super().__init__()
        with open('/code/MTST_ECE/config.yaml', 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.base_encoder = BertEncoder(self.cfg['pretrained_bert_name'])
        self.sl_model = SLModel(self.cfg)

    def construct(self, ids_padding_tensor, mask_tensor, document_len, tag_labels, mode):
        pooled, clause_state_list = self.base_encoder(ids_padding_tensor, mask_tensor, document_len)
        return self.sl_model(clause_state_list, tag_labels, mode)
