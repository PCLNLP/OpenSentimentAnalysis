from pathlib import Path
import yaml
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np

from MTST_ECE.model import Model
from MTST_ECE.utils.PrepareData import convert_document_to_ids, DataLoader, PrintMsg, Transform2Label
from MTST_ECE.eval import EvalEngine
from cybertron import BertAdam, BertTokenizer


class CEWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(CEWithLossCell, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.cast_op = ops.Cast()

    def construct(self, ids_padding_tensor, mask_tensor, document_len, tag_labels, emo_labels, cau_labels, mode):
        out = self._backbone(ids_padding_tensor, mask_tensor, document_len, tag_labels, mode)
        tag_log_probs = out[0]
        emo_log_probs = out[1]
        cau_log_probs = out[2]
        tag_labels_tensor = Tensor(tag_labels, mindspore.int32)
        emo_labels_tensor = Tensor(emo_labels, mindspore.int32)
        cau_labels_tensor = Tensor(cau_labels, mindspore.int32)
        tag_loss = self._loss_fn(tag_log_probs, tag_labels_tensor, Tensor(np.ones(tag_log_probs.shape[1]).astype(np.float32)))
        emo_loss = self._loss_fn(emo_log_probs, emo_labels_tensor, Tensor(np.ones(emo_log_probs.shape[1]).astype(np.float32)))
        cau_loss = self._loss_fn(cau_log_probs, cau_labels_tensor, Tensor(np.ones(cau_log_probs.shape[1]).astype(np.float32)))
        tag_loss = tag_loss[0]
        emo_loss = emo_loss[0]
        cau_loss = cau_loss[0]
        loss = 0.50*tag_loss + 0.25*emo_loss + 0.25*cau_loss
        return loss


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        opt.data_dir = Path(opt.data_dir)
        self.trainset, self.validset, self.testset, _ = DataLoader(None, 'old', opt.data_dir / opt.dataset, None)
        self.net = Model()
        with open('MTST_ECE/config.yaml', 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        bert_trainable = self.net.base_encoder.trainable_params()
        sl_trainable = self.net.sl_model.trainable_params()
        optimizer_parameters = [
            {'params': [p for p in sl_trainable if len(p.shape) > 1], 'weight_decay': opt.weight_decay},
            {'params': [p for p in sl_trainable if len(p.shape) == 1], 'weight_decay': 0.0},
            {'params': bert_trainable, 'lr': 1e-5}]
        train_iter_len = (len(self.trainset[0]) // opt.batch_size) + 1
        self.optimizer = BertAdam(optimizer_parameters, lr=opt.lr, warmup=opt.warmup,
                                  t_total=train_iter_len * opt.num_epochs)
        self.loss_fn = ops.NLLLoss()
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
                doc_len_list = [len(x.split('\x01')) for x in document_list]
                tag_labels, emo_labels, cau_labels = Transform2Label(self.trainset[1][start:end], doc_len_list, self.cfg['scope'])
                ids_padding_tensor, mask_tensor, document_len = convert_document_to_ids(document_list)
                loss = train_net(ids_padding_tensor, mask_tensor, document_len, tag_labels,
                                 emo_labels, cau_labels, 1)
                batch_i += 1
                total_batch += 1
                print(f"train batch: {total_batch}    loss: {loss}")
                if total_batch % self.opt.log_step == 0:
                    eval_engine = EvalEngine(self.net)
                    valid_emo, valid_cau, valid_pair = eval_engine.eval(self.validset, self.opt.batch_size, self.cfg['scope'])
                    if valid_pair[2] > best_f1:
                        early_stop = 0
                        best_f1 = valid_pair[2]
                        best_batch = total_batch
                        print('*' * 50 + 'the performance in valid set...' + '*' * 50)
                        PrintMsg(total_batch, valid_emo, valid_cau, valid_pair)
                        mindspore.save_checkpoint(self.net, self.opt.save_ckpt_path)
            early_stop += 1
            if early_stop >= self.opt.patience or i_epoch == self.opt.num_epochs - 1:
                mindspore.load_param_into_net(self.net, mindspore.load_checkpoint(self.opt.save_ckpt_path))
                print('=' * 50 + 'the performance in test set...' + '=' * 50)
                eval_engine = EvalEngine(self.net)
                test_emo, test_cau, test_pair = eval_engine.eval(self.testset, self.opt.batch_size, self.cfg['scope'])
                PrintMsg(best_batch, test_emo, test_cau, test_pair)
                break

    def eval(self, ckpt):
        mindspore.load_param_into_net(self.net, mindspore.load_checkpoint(ckpt))
        print('=' * 50 + 'the performance in test set...' + '=' * 50)
        eval_engine = EvalEngine(self.net)
        test_emo, test_cau, test_pair = eval_engine.eval(self.testset, self.opt.batch_size, self.cfg['scope'])
        PrintMsg('*', test_emo, test_cau, test_pair)
