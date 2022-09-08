from pathlib import Path

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from cybertron import BertAdam, BertTokenizer

from Trans_ECE.model import TransModel
from Trans_ECE.utils.PrepareData import DataLoader, PrintMsg
from Trans_ECE.utils.PrepareData import convert_document_to_ids
from Trans_ECE.utils.Transform import Text2ActionSequence, Text2SingleLabel
from Trans_ECE.eval import EvalEngine


class CEWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(CEWithLossCell, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.cast_op = ops.Cast()

    def construct(self, ids_padding_tensor, mask_tensor, document_len, single_labels, action_sequence_list, mode):
        out = self._backbone(ids_padding_tensor, mask_tensor, document_len, single_labels, action_sequence_list, mode)
        single_loss = self._loss_fn(out[0], out[1])
        tuple_loss = self._loss_fn(out[2], out[3])
        loss = single_loss + tuple_loss
        return loss


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        opt.data_dir = Path(opt.data_dir)
        self.trainset, self.validset, self.testset, _ = DataLoader(None, 'old', opt.data_dir / opt.dataset, None)
        self.train_action_sequence = Text2ActionSequence(self.trainset)
        self.single_labels_list = Text2SingleLabel(self.trainset)
        self.net = TransModel()
        bert_trainable = self.net.base_encoder.trainable_params()
        trans_trainable = self.net.trans_model.trainable_params()
        optimizer_parameters = [
            {'params': [p for p in trans_trainable if len(p.shape) > 1], 'weight_decay': opt.weight_decay},
            {'params': [p for p in trans_trainable if len(p.shape) == 1], 'weight_decay': 0.0},
            {'params': bert_trainable, 'lr': 1e-5}]
        train_iter_len = (len(self.trainset[0]) // opt.batch_size) + 1
        self.optimizer = BertAdam(optimizer_parameters, lr=opt.lr, warmup=opt.warmup,
                                  t_total=train_iter_len * opt.num_epochs)
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction='mean', sparse=True)
        self.tokenizer = BertTokenizer.load(opt.bert_tokenizer)
        self.ckpt_parent = Path(self.opt.save_ckpt_path).parent
        if not self.ckpt_parent.exists():
            self.ckpt_parent.mkdir(parents=True, exist_ok=True)

    def train(self):
        total_batch, early_stop = 0, 0
        best_batch, best_f1 = 0, 0.0
        loss_net = CEWithLossCell(self.net, self.loss_fn)
        train_net = nn.TrainOneStepCell(loss_net, optimizer=self.optimizer)
        for i_epoch in range(self.opt.num_epochs):
            print(f'[EPOCH] epoch {i_epoch + 1}/{self.opt.num_epochs}', flush=True)
            batch_i = 0
            while batch_i * self.opt.batch_size < len(self.trainset[0]):
                train_net.set_train(True)
                start, end = batch_i * self.opt.batch_size, (batch_i + 1) * self.opt.batch_size
                document_list = self.trainset[0][start: end]
                ids_padding_tensor, mask_tensor, document_len = convert_document_to_ids(document_list)
                action_sequence_list = self.train_action_sequence[start: end]
                single_labels = self.single_labels_list[start: end]
                loss = train_net(Tensor(ids_padding_tensor), Tensor(mask_tensor), document_len, single_labels,
                                 action_sequence_list, 1)
                batch_i += 1
                total_batch += 1
                print(f"train batch: {total_batch}    loss: {loss}")
                if total_batch % self.opt.log_step == 0:
                    eval_engine = EvalEngine(self.net)
                    valid_emo_metric, valid_cse_metric, valid_pr_metric = eval_engine.eval(self.validset,
                                                                                           self.opt.batch_size)
                    if valid_pr_metric[2] > best_f1:
                        early_stop = 0
                        best_f1 = valid_pr_metric[2]
                        best_batch = total_batch
                        print('*' * 50 + 'the performance in valid set...' + '*' * 50)
                        PrintMsg(total_batch, valid_emo_metric, valid_cse_metric, valid_pr_metric)
                        mindspore.save_checkpoint(self.net, self.opt.save_ckpt_path)
            early_stop += 1
            if early_stop >= self.opt.patience or i_epoch == self.opt.num_epochs - 1:
                mindspore.load_param_into_net(self.net, mindspore.load_checkpoint(self.opt.save_ckpt_path))
                print('=' * 50 + 'the performance in test set...' + '=' * 50)
                eval_engine = EvalEngine(self.net)
                test_emo_metric, test_cse_metric, test_pr_metric = eval_engine.eval(self.testset, self.opt.batch_size)
                PrintMsg(best_batch, test_emo_metric, test_cse_metric, test_pr_metric)
                pre, rec, f1 = test_pr_metric[0], test_pr_metric[1], test_pr_metric[2]
                print(f'Test Results:\npre: {pre}\trec: {rec}\tf1: {f1}')
                break

    def eval(self, ckpt):
        mindspore.load_param_into_net(self.net, mindspore.load_checkpoint(ckpt))
        print('=' * 50 + 'the performance in test set...' + '=' * 50)
        eval_engine = EvalEngine(self.net)
        test_emo_metric, test_cse_metric, test_pr_metric = eval_engine.eval(self.testset, self.opt.batch_size)
        PrintMsg('*', test_emo_metric, test_cse_metric, test_pr_metric)
        pre, rec, f1 = test_pr_metric[0], test_pr_metric[1], test_pr_metric[2]
        print(f'Test Results:\npre: {pre}\trec: {rec}\tf1: {f1}')
