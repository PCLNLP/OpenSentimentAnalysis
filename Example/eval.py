import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


class EvalEngine:
    def __init__(self, model):
        self.model = model
        self.f1 = nn.F1()
        self.concat = ops.Concat(axis=0)

    def eval(self, dataset):
        self.model.set_train(False)
        t_targets_all, t_outputs_all = None, None
        for t_sample_batched in dataset:
            t_targets = mindspore.Tensor(t_sample_batched['polarity'])
            t_outputs = self.model(
                mindspore.Tensor(t_sample_batched['text_indices']),
                mindspore.Tensor(t_sample_batched['entity_graph']),
                mindspore.Tensor(t_sample_batched['attribute_graph']),
                mindspore.Tensor(t_sample_batched['seq_length']),
            )
            if t_targets_all is None:
                t_targets_all = t_targets
                t_outputs_all = t_outputs
            else:
                t_targets_all = self.concat((t_targets_all, t_targets))
                t_outputs_all = self.concat((t_outputs_all, t_outputs))
        n_test_correct = int(sum((t_outputs_all.argmax(-1) == t_targets_all)))
        n_test_total = len(t_outputs_all)
        acc = n_test_correct / n_test_total
        f1 = self._f1(t_outputs_all, t_targets_all)
        self.model.set_train(True)
        return acc, f1

    def _f1(self, a, b):
        self.f1.clear()
        self.f1.update(a, b)
        return self.f1.eval(average=True)

    def update_wight(self, weight_path):
        weight = mindspore.load_checkpoint(weight_path)
        mindspore.load_param_into_net(self.model, weight)


# def test_eval():
#     opt = parse_args()
#     context.set_context(
#         mode=context.GRAPH_MODE,
#         device_target=opt.device,
#         device_id=opt.device_id,
#     )
#     if opt.save_graphs:
#         context.set_context(save_graphs=True, save_graphs_path="./_save_{}".format(int(time.time())))
#     begin = time.time()
#     _, val_dataset, test_dataset, embedding_matrix = build_dataset(
#         dataset_prefix=opt.dataset_prefix,
#         knowledge_base=opt.knowledge_base,
#         worker_num=opt.worker_num,
#         valset_ratio=opt.valset_ratio
#     )
#
#     val_dataset = test_dataset.create_dict_iterator(num_epochs=1)
#     t2 = time.time()
#     print('load data:', t2 - begin)
#     # import numpy
#     # embedding_matrix = numpy.load('data/com_embedding_matrix.npy', allow_pickle=True)
#     net = AAGCN(embedding_matrix, opt)
#
#     # import torch
#     # torch_model = torch.load('weight/aagcn_15_rest_senticnet.pkl')
#     # keys = torch_model.keys()
#     # m2t_map = {
#     #     'embed.embedding_table':'embed.weight'
#     # }
#     # for p in net.get_parameters():
#     #     if p.name in keys:
#     #         data = torch_model[p.name].numpy()
#     #     elif p.name in m2t_map:
#     #         data = torch_model[m2t_map[p.name]].numpy()
#     #     else:
#     #         print(p.name)
#     #     p.set_data(Tensor(data, dtype=p.dtype))
#
#     if os.path.exists(opt.pretrained):
#         param_dict = load_checkpoint(opt.pretrained)
#         load_param_into_net(net, param_dict)
#         for m in net.get_parameters():
#             print(m)
#
#     engine = EvalEngine(net)
#     print(engine.eval(val_dataset))

# if __name__=='__main__':
#     test_eval()