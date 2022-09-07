import math
import yaml

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
from cybertron import BertModel
from mindspore.common.initializer import initializer, HeUniform, Uniform, Normal, _calculate_fan_in_and_fan_out
from Trans_ECE.utils.Transform import action2id
from Trans_ECE.utils.init import XavierNormal


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


class TransitionModel(nn.Cell):

    def __init__(self, config):
        super().__init__()
        self.is_bi = config['is_bi']
        self.bert_output_size = config['bert_output_size']
        self.mlp_size = config['mlp_size']
        self.cell_size = config['cell_size']
        self.operation_type = config['operation_type']
        self.scale_factor = config['scale_factor']
        self.dropout = config['dropout']
        self.layers = config['layers']
        self.max_document_len = config['max_document_len']
        self.position_ebd_dim = config['position_ebd_dim']
        self.position_embedding = Embedding(self.max_document_len - 1, self.position_ebd_dim)
        self.position_trainable = config['position_trainable']
        self.action_ebd_dim = config['action_ebd_dim']
        self.action_type_num = config['action_type_num']
        self.action_embedding = Embedding(self.action_type_num, self.action_ebd_dim)
        self.action_trainable = config['action_trainable']
        self.label_num = config['label_num']
        self.stack_cell = nn.LSTM(self.bert_output_size, self.cell_size, self.layers, bidirectional=self.is_bi)
        self.buffer_cell = nn.LSTM(self.bert_output_size, self.cell_size, self.layers, bidirectional=self.is_bi)
        self.action_cell = nn.LSTM(self.action_ebd_dim, self.cell_size, self.layers, bidirectional=False)

        if self.operation_type == 'attention':
            self.attention_layer = nn.SequentialCell(
                Dense(self.bert_output_size, self.hidden_size),
                Dense(self.hidden_size, 1)
            )

        # The classifier for the CA action
        self.single_MLP = nn.SequentialCell(
            Dense(self.bert_output_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(alpha=0.01),
            nn.Dropout(1 - self.dropout),
            Dense(self.mlp_size, self.mlp_size // self.scale_factor),
            nn.BatchNorm1d(self.mlp_size // self.scale_factor),
            nn.LeakyReLU(alpha=0.01),
            nn.Dropout(1 - self.dropout),
            Dense(self.mlp_size // self.scale_factor, 2)
        )

        # The classifier for the other actions
        self.tuple_MLP = nn.SequentialCell(
            Dense(self.cell_size * 2 * 2 + self.cell_size + self.position_ebd_dim, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(alpha=0.01),
            nn.Dropout(1 - self.dropout),
            Dense(self.mlp_size, self.mlp_size // self.scale_factor),
            nn.BatchNorm1d(self.mlp_size // self.scale_factor),
            nn.LeakyReLU(alpha=0.01),
            nn.Dropout(1 - self.dropout),
            Dense(self.mlp_size // self.scale_factor, self.label_num)
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
        self.position_embedding.embedding_table.requires_grad = self.position_trainable
        self.action_embedding.embedding_table.requires_grad = self.action_trainable

    def init_hidden(self, batch_size, mode):
        if mode == 'action':
            hidden = (Tensor(initializer('zeros', (self.layers, batch_size, self.cell_size))),
                      Tensor(initializer('zeros', (self.layers, batch_size, self.cell_size))))
        else:
            if self.is_bi:
                hidden = (Tensor(initializer('zeros', (self.layers * 2, batch_size, self.cell_size))),
                          Tensor(initializer('zeros', (self.layers * 2, batch_size, self.cell_size)))
                          )
            else:
                hidden = (Tensor(initializer('zeros', (self.layers, batch_size, self.cell_size))),
                          Tensor(initializer('zeros', (self.layers, batch_size, self.cell_size)))
                          )
        return hidden

    def operation(self, state_1, state_2, state_3):
        if self.operation_type == 'concatenate':
            inputs = ops.Concat()([state_1, state_2, state_3])
        elif self.operation_type == 'mean':
            inputs = (state_1 + state_2 + state_3) / 3.
        elif self.operation_type == 'sum':
            inputs = state_1 + state_2 + state_3
        elif self.operation_type == 'attention':
            stack_state = ops.Stack()([state_1, state_2, state_3])
            attention_logits = self.attention_layer(stack_state)
            attention_weights = ops.Softmax(0)(attention_logits)
            inputs = ops.Squeeze(1)(ops.MatMul()(ops.Transpose()(stack_state, (1, 0)), attention_weights))
        else:
            print('operation type error!')
        return inputs

    def action_encoder(self, action_sequence_list):
        action_list = [[x[-1] for x in asl] for asl in action_sequence_list]
        action_len_list = [len(x) for x in action_list]
        max_action_len = max(action_len_list)
        action_padding_list = [[5] + x[:-1] + [6] * (max_action_len - len(x)) for x in action_list]
        action_padding_tensor = Tensor(action_padding_list)

        inputs = ops.Transpose()(self.action_embedding(action_padding_tensor), (1, 0, 2))
        bs = inputs.shape[1]
        init_state = self.init_hidden(bs, 'action')
        outputs, _ = self.action_cell(inputs, init_state)
        outputs_permute = ops.Transpose()(outputs, (1, 0, 2))
        output_list = [outputs_permute[i][:al] for i, al in enumerate(action_len_list)]
        output_stack = ops.Concat()(output_list)
        return output_stack

    @staticmethod
    def reversal_sample(sk_1, sk_2, action):
        sk_1_construct, sk_1_backward = ops.Split(axis=0, output_num=2)(sk_1)
        sk_2_construct, sk_2_backward = ops.Split(axis=0, output_num=2)(sk_2)
        ori_sk_1, ori_sk_2, ori_act = sk_1_construct, sk_2_construct, action
        rev_sk_1, rev_sk_2 = sk_2_backward, sk_1_backward
        if action == action2id['shift']:
            rev_act = action2id['shift']
        elif action == action2id['right_arc_ln']:
            rev_act = action2id['left_arc_ln']
        elif action == action2id['right_arc_lt']:
            rev_act = action2id['left_arc_lt']
        elif action == action2id['left_arc_ln']:
            rev_act = action2id['right_arc_ln']
        elif action == action2id['left_arc_lt']:
            rev_act = action2id['right_arc_lt']
        return ori_sk_1, ori_sk_2, ori_act, rev_sk_1, rev_sk_2, rev_act

    def train_mode(self, clause_state_list, action_sequence_list):
        tuple_labels_list, distance_list = [], []
        sk_input_list, bf_input_list, sk_len_list, bf_len_list = [], [], [], []
        for d_i in range(len(clause_state_list)):
            clause_state, action_sequence = clause_state_list[d_i], action_sequence_list[d_i]
            for a_s in action_sequence:
                stack, buffer, action = a_s[0], a_s[1], a_s[2]
                tuple_labels_list.append(action)
                stack_input = ops.Stack()([clause_state[s] for s in stack])
                sk_len_list.append(stack_input.shape[0])
                sk_input_list.append(stack_input)
                distance_list.append(int(abs(stack[-2] - stack[-1])))
                if len(buffer) > 0:
                    buffer_input = ops.Stack()([clause_state[b] for b in buffer])
                else:
                    if stack[-1] < clause_state.shape[0] - 1:
                        buffer_input = ops.Stack()([clause_state[stack[-1] + 1]])
                    else:
                        buffer_input = ops.Stack()([clause_state[stack[-1]]])
                bf_input_list.append(buffer_input)
                bf_len_list.append(buffer_input.shape[0])
        max_sk_len, max_bf_len = max(sk_len_list), max(bf_len_list)
        tmp_sk_list, tmp_bf_list = [], []
        for sk_input, bf_input in zip(sk_input_list, bf_input_list):
            sk_row, sk_column = sk_input.shape
            bf_row, bf_column = bf_input.shape
            sk_tmp = mindspore.Parameter(initializer('zeros', (max_sk_len, sk_column)))
            bf_tmp = mindspore.Parameter(initializer('zeros', (max_bf_len, bf_column)))
            sk_tmp[:sk_row] = sk_input
            bf_tmp[:bf_row] = bf_input
            tmp_sk_list.append(sk_tmp)
            tmp_bf_list.append(bf_tmp)
        sk_input_tensor, bf_input_tensor = ops.Transpose()(ops.Stack()(tmp_sk_list), (1, 0, 2)), ops.Transpose()(
            ops.Stack()(tmp_bf_list), (1, 0, 2))
        sk_bs, bf_bs = sk_input_tensor.shape[1], bf_input_tensor.shape[1]
        sk_init, bf_init = self.init_hidden(sk_bs, 'else'), self.init_hidden(bf_bs, 'else')
        sk_output, _ = self.stack_cell(sk_input_tensor, sk_init)
        bf_output, _ = self.buffer_cell(bf_input_tensor, bf_init)
        sk_output_permute, bf_output_permute = ops.Transpose()(sk_output, (1, 0, 2)), ops.Transpose()(bf_output,
                                                                                                      (1, 0, 2))
        del sk_output
        del bf_output
        sk_update_list = [sk_output_permute[i][:sk_len] for i, sk_len in enumerate(sk_len_list)]
        bf_update_list = [bf_output_permute[i][:bf_len] for i, bf_len in enumerate(bf_len_list)]
        final_inputs_list, final_labels_list, final_distance_list, final_action_output = [], [], [], []
        inx = 0
        action_output = self.action_encoder(action_sequence_list)
        for sk_update, bf_update in zip(sk_update_list, bf_update_list):
            action = tuple_labels_list[inx]
            ori_sk_1, ori_sk_2, ori_act, rev_sk_1, rev_sk_2, rev_act = self.reversal_sample(sk_update[-2],
                                                                                            sk_update[-1], action)
            ori_inputs = self.operation(ori_sk_1, ori_sk_2, bf_update[0])
            final_inputs_list.append(ori_inputs)
            final_labels_list.append(ori_act)
            final_distance_list.append(distance_list[inx])
            final_action_output.append(action_output[inx])
            rev_inputs = self.operation(rev_sk_1, rev_sk_2, bf_update[0])
            final_inputs_list.append(rev_inputs)
            final_labels_list.append(rev_act)
            final_distance_list.append(distance_list[inx])
            final_action_output.append(action_output[inx])
            inx += 1
        del sk_update_list
        del bf_update_list
        distance_tensor = Tensor(final_distance_list)
        pos_embedding = self.position_embedding(distance_tensor)
        tuple_inputs_tensor = ops.Concat(axis=1)(
            [ops.Stack()(final_inputs_list), ops.Stack()(final_action_output), pos_embedding])

        tuple_labels_tensor = Tensor(final_labels_list, mstype.int64)
        tuple_logits = self.tuple_MLP(tuple_inputs_tensor)

        return tuple_logits, tuple_labels_tensor

    def predict_action(self, state, stack, buffer, action, act_hidden):
        stack_input = ops.ExpandDims()(ops.Stack()([state[s] for s in stack]), 0)
        sk_init_state = self.init_hidden(1, 'else')
        sk_output, _ = self.stack_cell(ops.Transpose()(stack_input, (1, 0, 2)), sk_init_state)
        sk_output_permute = ops.Squeeze(0)(ops.Transpose()(sk_output, (1, 0, 2)))
        if len(buffer) > 0:
            buffer_input = ops.ExpandDims()(ops.Stack()([state[b] for b in buffer]), 0)
        else:
            if stack[-1] < state.shape[0] - 1:
                buffer_input = ops.ExpandDims()(ops.Stack()([state[stack[-1] + 1]]), 0)
            else:
                buffer_input = ops.ExpandDims()(ops.Stack()([state[stack[-1]]]), 0)
        bf_init_state = self.init_hidden(1, 'else')
        bf_output, _ = self.buffer_cell(ops.Transpose()(buffer_input, (1, 0, 2)), bf_init_state)
        bf_output_permute = ops.Squeeze(0)(ops.Transpose()(bf_output, (1, 0, 2)))

        act_input = self.action_embedding(Tensor([[action]]))
        act_output, act_hidden = self.action_cell(act_input, act_hidden)
        act_output_permute = ops.Squeeze(0)(ops.Squeeze(0)(act_output))

        change_1_construct, change_1_backward = ops.Split(output_num=2)(sk_output_permute[-2])
        change_2_construct, change_2_backward = ops.Split(output_num=2)(sk_output_permute[-1])

        c_inputs = self.operation(change_1_construct, change_2_construct, bf_output_permute[0])
        distance = Tensor(int(abs(stack[-2] - stack[-1])))
        pos_embedding = self.position_embedding(distance)
        inputs = ops.ExpandDims()(ops.Concat()([c_inputs, act_output_permute, pos_embedding]), 0)
        tuple_logits = self.tuple_MLP(inputs)
        tuple_probs = ops.Softmax(axis=1)(tuple_logits)
        action = tuple_probs.argmax(1).asnumpy()[0]
        return action, act_hidden

    def eval_mode(self, clause_state_list):
        predicts = []
        batch_size = len(clause_state_list)
        for d_i in range(batch_size):
            preds, stack = [], []
            document_len = clause_state_list[d_i].shape[0]
            buffer = list(range(document_len))
            stack.append(0), stack.append(1)
            buffer.remove(0), buffer.remove(1)
            state = clause_state_list[d_i]
            action = 5
            act_hidden = self.init_hidden(1, 'action')
            while len(buffer) > 0:
                if len(stack) < 2:
                    stack.append(buffer.pop(0))
                action, act_hidden = self.predict_action(state, stack, buffer, action, act_hidden)
                if action == action2id['shift']:
                    if len(buffer) > 0:
                        stack.append(buffer.pop(0))
                elif action == action2id['right_arc_ln']:
                    preds.append((stack[-1],))
                    stack.pop(-2)
                elif action == action2id['right_arc_lt']:
                    preds.append((stack[-1], stack[-2]))
                    stack.pop(-2)
                elif action == action2id['left_arc_ln']:
                    preds.append((stack[-2],))
                    if len(buffer) > 0:
                        stack.append(buffer.pop(0))
                else:  # left_arc_lt
                    preds.append((stack[-2], stack[-1]))
                    stack.pop(-1)

            while len(stack) >= 2:
                action, act_hidden = self.predict_action(state, stack, buffer, action, act_hidden)

                if action == action2id['right_arc_ln']:
                    preds.append((stack[-1],))
                    stack.pop(-2)
                elif action == action2id['right_arc_lt']:
                    preds.append((stack[-1], stack[-2]))
                    stack.pop(-2)
                elif action == action2id['left_arc_ln']:
                    preds.append((stack[-2],))
                    stack.pop(-1)
                elif action == action2id['left_arc_lt']:
                    preds.append((stack[-2], stack[-1]))
                    stack.pop(-1)
                else:
                    break

            unique_preds = []
            for pd in preds:
                if pd not in unique_preds:
                    unique_preds.append(pd)
            predicts.append(unique_preds)

        return predicts

    def construct(self, pooled, single_labels_list, clause_state_list, action_sequence_list, mode):
        if mode == 1:
            single_logits = self.single_MLP(pooled)
            single_labels_tensor = Tensor([i for x in single_labels_list for i in x])
            tuple_logits, tuple_labels_tensor = self.train_mode(clause_state_list, action_sequence_list)
            return single_logits, single_labels_tensor, tuple_logits, tuple_labels_tensor
        if mode == 0:
            single_logits = self.single_MLP(pooled)
            single_preds = list(ops.Softmax(axis=1)(single_logits).argmax(1).asnumpy())
            tuple_preds = self.eval_mode(clause_state_list)
            return single_preds, tuple_preds
        else:
            print('mode error!')


class TransModel(nn.Cell):
    def __init__(self):
        super().__init__()
        with open('Trans_ECE/config.yaml', 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.base_encoder = BertEncoder(self.cfg['pretrained_bert_name'])
        self.trans_model = TransitionModel(self.cfg)

    def construct(self, ids_padding_tensor, mask_tensor, document_len, single_labels_list, action_sequence_list, mode):
        pooled, clause_state_list = self.base_encoder(ids_padding_tensor, mask_tensor, document_len)
        return self.trans_model(pooled, single_labels_list, clause_state_list, action_sequence_list, mode)
