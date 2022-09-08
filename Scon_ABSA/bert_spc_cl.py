import math
import yaml
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import initializer, HeUniform, Uniform, Normal, _calculate_fan_in_and_fan_out
from cybertron import BertModel
from Scon_ABSA.utils.init import XavierNormal

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

class BERT_SPC_CL(nn.Cell):
    def __init__(self):
        super(BERT_SPC_CL, self).__init__()
        with open('/code/Scon_ABSA/config.yaml', 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.bert = BertModel.load(self.cfg['pretrained_bert_name'])
        self.dropout = nn.Dropout(self.cfg['dropout'])
        self.dense = Dense(self.cfg['bert_dim'], self.cfg['polarities_dim'])
        self.dense2 = Dense(self.cfg['bert_dim'], 2)

    def reset_parameters(self):
        for cell in self.cells():
            if type(cell) != BertModel:
                for param in self.get_parameters():
                    if param.requires_grad:
                        if len(param.shape) > 1:
                            param.set_data(initializer(XavierNormal(), param.shape))
                        else:
                            stdv = 1. / math.sqrt(param.shape[0])
                            param.set_data(initializer(Uniform(stdv), param.shape))

    def construct(self, inputs):
        text_bert_indices = inputs[self.cfg['input_columns'][0]]
        bert_segments_ids = inputs[self.cfg['input_columns'][1]]
        text_embed, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)        
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        logits2 = self.dense2(pooled_output)
        return logits,pooled_output,logits2