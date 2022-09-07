import time
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from collections import namedtuple
from mindspore.common import mutable

class EvalEngine:
    def __init__(self, model):
        self.model = model
        self.model.set_train(False)
        self.f1 = nn.F1()
        self.acc = nn.Accuracy()
        self.concat = ops.Concat(axis=0)
    
    def eval(self, dataset):    
        t_targets_all, t_outputs_all = None, None
        eval_start_time = time.time()
        eval_steps = 0
        for batch in dataset.create_dict_iterator():
            #breakpoint()
            batch = mutable(batch)
            print(f'Eval step: {eval_steps}')
            t_targets = batch['polarity']
            t_outputs = self.model(batch)[0]

            if t_targets_all is None:
                t_targets_all = t_targets
                t_outputs_all = t_outputs
            else:
                t_targets_all = self.concat((t_targets_all, t_targets))
                t_outputs_all = self.concat((t_outputs_all, t_outputs))
            eval_steps += 1
        eval_end_time = time.time()
        print(f'eval_speed: {(eval_end_time - eval_start_time) / eval_steps} s/step')
        acc = self._acc(t_outputs_all, t_targets_all)
        f1 = self._f1(t_outputs_all, t_targets_all)
        
        Results = namedtuple('Results', ['acc', 'f1'])
        results = Results(acc, f1)
        for i in range(len(results)):
            print(f'{results._fields[i]}: {results[i]}')
        return results

    def _f1(self, a, b):
        self.f1.clear()
        self.f1.update(a, b)
        return self.f1.eval(average=True)

    def _acc(self, a, b):
        self.acc.clear()
        self.acc.update(a, b)
        return self.acc.eval()

    def update_wight(self, weight_path):
        weight = mindspore.load_checkpoint(weight_path)
        mindspore.load_param_into_net(self.model, weight)

