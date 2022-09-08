import time
from pathlib import Path

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import mutable

from SenticBERT_ABSA.model import Model
from SenticBERT_ABSA.dataset import build_dataset
from SenticBERT_ABSA.eval import EvalEngine


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.net = Model()
        self.trainset, self.testset = build_dataset(opt)
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.optimizer = nn.Adam(self.net.trainable_params(), opt.lr, weight_decay=opt.weight_decay)
        self.grad_fn = ops.value_and_grad(self.forward_fn, None, self.net.trainable_params(), has_aux=True)
        self.ckpt_parent = Path(self.opt.save_ckpt_path).parent
        if not self.ckpt_parent.exists():
            self.ckpt_parent.mkdir(parents=True, exist_ok=True)

    def forward_fn(self, inputs, targets):
        logits = self.net(inputs)
        loss = self.loss_fn(logits, targets)
        return loss, logits

    def train_step(self, inputs, targets):
        (loss, logits), grads = self.grad_fn(inputs, targets)
        self.optimizer(grads)
        return loss, logits

    def train(self):
        best_res, best_epoch, global_step = 0, 0, 0
        for i_epoch in range(self.opt.num_epochs):
            self.net.set_train(True)
            print(f'[EPOCH] epoch {i_epoch + 1}/{self.opt.num_epochs}', flush=True)
            epoch_begin_time, step_accumulate_time = time.time(), 0
            for batch in self.trainset.create_dict_iterator():
                batch = mutable(batch)
                global_step += 1
                step_begin_time = time.time()
                loss, logits = self.train_step(batch, batch['polarity'])
                step_accumulate_time += (time.time() - step_begin_time)
                if global_step % self.opt.log_step == 0:
                    print(f'step: {global_step}, train_loss: {loss}, each_step_spend: {step_accumulate_time / self.opt.log_step:.2f}')
                    step_accumulate_time = 0
            print(f'[EPOCH] epoch {i_epoch + 1}/{self.opt.num_epochs} finished, each_epoch_spend: {time.time() - epoch_begin_time:.2f}')
            # eval
            eval_engine = EvalEngine(self.net)
            # results is a namedtuple with the first item as the main metric
            results = eval_engine.eval(self.testset)
            if results[0] > best_res:
                best_res = results[0]
                best_epoch = i_epoch
                mindspore.save_checkpoint(self.net, str(self.opt.save_ckpt_path))
            if i_epoch - best_epoch >= self.opt.patience:
                print('>>Early Stop!')
                break

    def eval(self, ckpt):
        eval_engine = EvalEngine(self.net)
        eval_engine.update_wight(ckpt)
        eval_engine.eval(self.testset)
