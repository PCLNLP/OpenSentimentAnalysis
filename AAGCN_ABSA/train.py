# -*- coding: utf-8 -*-
import numpy
import time
from collections import namedtuple

import mindspore
import mindspore.nn as nn
from mindspore import Tensor, Model, save_checkpoint
from mindspore.train.callback import Callback

from AAGCN_ABSA.dataset import build_dataset
from AAGCN_ABSA.model import AAGCN
from AAGCN_ABSA.utils.tools import print_args
from AAGCN_ABSA.eval import EvalEngine


class EvalCallBack(Callback):
    def __init__(self, net, model, opt, val_dataset, test_dataset):
        self.model = model
        self.opt = opt
        self.cur_test_epoch = 1
        self.eval_engine = EvalEngine(net)
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.save_ckpt_path = self.opt.data_dir / 'checkpoints' / self.opt.dataset / 'best_eval.ckpt'
        if not self.save_ckpt_path.exists():
            self.save_ckpt_path.mkdir(parents=True, exist_ok=True)

        self.best_step = 0
        self.best_model_acc = 0
        self.best_model_f1 = 0
        self.sum_step_spend = 0

    def val_best_model(self):
        self.eval_engine.update_wight(str(self.save_ckpt_path))
        val_acc, val_f1 = self.eval_engine.eval(self.test_dataset)
        print('>>> seed : {}, algo: {}, dataset: {}'.format(
            self.opt.seed, self.opt.algo, self.opt.dataset))
        print('>>> save: {}'.format(self.save_ckpt_path))
        print('>>> VAL  best_model_acc: {:4f}, best_model_f1: {:4f} best_step: {}'.format(
            self.best_model_acc * 100, self.best_model_f1 * 100, self.best_step))
        print('>>> TEST best_model_acc: {:4f}, best_model_f1: {:4f} best_step: {}'.format(
            val_acc * 100, val_f1 * 100, self.best_step))

    def epoch_begin(self, run_context=None):
        cb_params = run_context.original_args()
        print('[EPOCH] epoch {}/{}'.format(cb_params.cur_epoch_num,
                                           cb_params.epoch_num), flush=True)
        self.epoch_begin_time = time.time()

    def epoch_end(self, run_context=None):
        cb_params = run_context.original_args()
        print('[EPOCH] epoch {}/{} finished, spend: {:6f}'.format(
            cb_params.cur_epoch_num, cb_params.epoch_num,
            time.time() - self.epoch_begin_time), flush=True)
        # 训练结束后进行测试精度
        if cb_params.cur_epoch_num == cb_params.epoch_num:
            self.val_best_model()

    def step_begin(self, run_context=None):
        self.step_begin_time = time.time()

    def _get_loss(self, cb_params):
        loss = cb_params.net_outputs
        if isinstance(loss, tuple):
            if isinstance(loss[0], Tensor):
                return loss[0].asnumpy()
        if isinstance(loss, Tensor):
            return numpy.mean(loss.asnumpy())

    def step_end(self, run_context=None):
        cb_params = run_context.original_args()
        self.loss = self._get_loss(cb_params)
        self.sum_step_spend += (time.time() - self.step_begin_time)
        if cb_params.cur_step_num % self.opt.log_step == 0:
            self.cur_test_epoch += 1
            val_acc, val_f1 = self.eval_engine.eval(self.val_dataset)
            ops = '-DROP'
            if val_acc > self.best_model_acc:
                self.best_model_acc = val_acc
            if val_f1 > self.best_model_f1:
                self.best_model_f1 = val_f1
                ops = '+SAVE'
                self.best_step = cb_params.cur_step_num
                save_checkpoint(cb_params.train_network, str(self.save_ckpt_path))
            print('loss: {:.4f}, val_acc: {:.4f}, val_f1: {:.4f}, spend: {:.4f}, {}'.format(
                self.loss, val_acc, val_f1, self.sum_step_spend / self.opt.log_step, ops), flush=True)
            self.sum_step_spend = 0


class NetWithLoss(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(NetWithLoss, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, *input):
        out = self._backbone(input[0], input[2], input[3], input[4])
        loss = self._loss_fn(out, input[1])
        return loss


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        # 创建数据集
        self.train_dataset, self.val_dataset, self.test_dataset, self.embedding_matrix = build_dataset(opt)
        print('train_dataset:', self.train_dataset.get_dataset_size())
        print('val_dataset:', self.val_dataset.get_dataset_size())
        print('test_dataset:', self.test_dataset.get_dataset_size())
        self.step_size = self.train_dataset.get_dataset_size()
        self.test_epoch = (self.step_size * opt.num_epochs) // opt.log_step
        self.val_dataset = self.val_dataset.create_dict_iterator(num_epochs=self.test_epoch)
        self.test_dataset = self.test_dataset.create_dict_iterator(num_epochs=1)
        # 初始化网络
        self.net = AAGCN(self.embedding_matrix)
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        self.opt_adm = nn.Adam(self.net.trainable_params(),
                               opt.lr, weight_decay=opt.weight_decay)

    def train(self):
        train_begin = time.time()
        print('train begin: {} epoch, {} step per epoch, {} eval'.format(
            self.opt.num_epochs, self.step_size, self.test_epoch), flush=True)
        train_net = NetWithLoss(self.net, self.loss_fn)
        scale_manager = mindspore.DynamicLossScaleManager(2 ** 24, 2, 100)
        print_args(self.net, self.opt)
        train_net.set_train(True)
        model = Model(train_net, optimizer=self.opt_adm, loss_scale_manager=scale_manager)
        callback = EvalCallBack(self.net, train_net, self.opt, self.val_dataset, self.test_dataset)
        # 训练
        model.train(self.opt.num_epochs, self.train_dataset,
                    callbacks=callback, dataset_sink_mode=False)
        print('train over, total spend:', time.time() - train_begin)

    def eval(self, ckpt):
        eval_engine = EvalEngine(self.net)
        eval_engine.update_wight(ckpt)
        acc, f1 = eval_engine.eval(self.test_dataset)
        Results = namedtuple('Results', ['acc', 'f1'])
        results = Results(acc, f1)
        for i in range(len(results)):
            print(f'{results._fields[i]}: {results[i]}')
        return results
