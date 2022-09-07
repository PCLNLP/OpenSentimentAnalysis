import time

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import mutable

from Scon_ABSA.bert_spc_cl import BERT_SPC_CL
from Scon_ABSA.dataset import build_dataset
from Scon_ABSA.eval import EvalEngine
from Scon_ABSA.losses import SupConLoss


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.net = BERT_SPC_CL()
        self.trainset, self.testset = build_dataset(opt)
        self.criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.contrastiveLoss = SupConLoss()
        self.optimizer = nn.Adam(self.net.trainable_params(), opt.lr, weight_decay=opt.weight_decay)
        self.grad_fn = ops.value_and_grad(self.forward_fn, None, self.net.trainable_params(), has_aux=True)
        self.cur_acc = nn.Accuracy()
        self.save_ckpt_path = self.opt.data_dir / 'checkpoints' / self.opt.dataset / 'best_eval.ckpt'
        if not self.save_ckpt_path.exists():
            self.save_ckpt_path.mkdir(parents=True, exist_ok=True)

    def forward_fn(self, inputs):
        outputs = self.net(inputs)
        logits = outputs[0]
        loss2 = self.contrastiveLoss(outputs[1], inputs['cllabel'])
        loss1 = self.criterion(outputs[0], inputs['polarity'])
        loss3 = self.contrastiveLoss(outputs[1], inputs['polabel'])
        loss = loss1.mean()+loss2+loss3
        return loss, logits

    #@ms_function
    def train_step(self, inputs):
        (loss, logits), grads = self.grad_fn(inputs)
        self.optimizer(grads)
        return loss, logits

    def train(self):
        best_res, best_epoch, global_step = 0, 0, 0
        for i_epoch in range(self.opt.num_epochs):
            self.net.set_train(True)
            print(f'[EPOCH] epoch {i_epoch + 1}/{self.opt.num_epochs}', flush=True)
            epoch_begin_time, step_accumulate_time = time.time(), 0
            self.cur_acc.clear()
            for batch in self.trainset.create_dict_iterator():
                batch = mutable(batch)
                global_step += 1
                step_begin_time = time.time()
                loss, logits = self.train_step(batch)
                step_accumulate_time += (time.time() - step_begin_time)
                if global_step % self.opt.log_step == 0:
                    self.cur_acc.update(logits, batch['polarity'])
                    print(f'step: {global_step}, train_loss: {loss}, train_acc: {self.cur_acc.eval():.4f}, each_step_spend: {step_accumulate_time / self.opt.log_step:4f}')
                    step_accumulate_time = 0
            print(f'[EPOCH] epoch {i_epoch + 1}/{self.opt.num_epochs} finished, each_epoch_spend: {time.time() - epoch_begin_time:.4f}')
            # eval
            eval_engine = EvalEngine(self.net)
            results = eval_engine.eval(self.testset)
            if results[0] > best_res:
                best_res = results[0]
                best_epoch = i_epoch
                mindspore.save_checkpoint(self.net, str(self.save_ckpt_path))
            if i_epoch - best_epoch >= self.opt.patience:
                print('>>Early Stop!')
                break

    def eval(self, ckpt):
        eval_engine = EvalEngine(self.net)
        eval_engine.update_wight(ckpt)
        eval_engine.eval(self.testset)
