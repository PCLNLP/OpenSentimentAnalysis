import argparse
import sys

def print_args(model, opt):
    print('===========================')
    print('[ ARGS ]')
    print(sys.argv)
    print('[ MODEL ]')
    print(model)
    print('[ PARAM ]')
    n_trainable_params, n_nontrainable_params = 0, 0
    for p in model.get_parameters():
        print(p)
        n_params = 1
        for i in p.shape:
            n_params*=i
        if p.requires_grad:
            n_trainable_params += n_params
        else:
            n_nontrainable_params += n_params
    print('[ARGS] n_trainable_params: {}, n_nontrainable_params: {}'.format(n_trainable_params, n_nontrainable_params), flush=True)
    print('[ARGS] training arguments:', flush=True)
    for arg in vars(opt):
        print('>>> {0}: {1}'.format(arg, getattr(opt, arg)), flush=True)
    print('============================')

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num_epoch', default=100, type=int)
#     parser.add_argument('--batch_size', default=16, type=int)
#     parser.add_argument('--worker_num', default=1, type=int)
#     parser.add_argument('--seed', default=1000, type=int)
#     parser.add_argument('--device', default='Ascend', type=str, choices=['Ascend', 'CPU', 'GPU'], help='device: [Ascend, CPU, GPU]')
#     parser.add_argument('--device_id', default=0, type=int)
#     parser.add_argument('--mode', default='feed', type=str)
#     parser.add_argument('--eval', default=False, type=bool)
#     parser.add_argument('--save_graphs', default=False, type=bool)
#     parser.add_argument('--amp', default=False, type=bool)
#     parser.add_argument('--profile', default=False, type=bool)
#
#     parser.add_argument('--model_name', default='aagcn', type=str)
#     parser.add_argument('--pretrained', default='', type=str)
#     parser.add_argument('--embed_dim', default=300, type=int)
#     parser.add_argument('--hidden_dim', default=300, type=int)
#     parser.add_argument('--polarities_dim', default=3, type=int)
#
#     parser.add_argument('--optimizer', default='adam', type=str)
#     parser.add_argument('--initializer', default='xavier_uniform_', type=str)
#     parser.add_argument('--learning_rate', default=0.001, type=float)
#     parser.add_argument('--l2reg', default=0.00001, type=float)
#
#     parser.add_argument('--knowledge_base', default='senticnet', type=str, help='conceptnet, senticnet')
#     parser.add_argument('--dataset_prefix', default='15_rest', type=str)
#     parser.add_argument('--valset_ratio', default=0.1, type=float)
#
#     parser.add_argument('--log_step', default=5, type=int)
#     parser.add_argument('--save_ckpt_path', default='ckpt/epoch', type=str)
#     parser.add_argument('--keep_ckpt_max', default=5, type=int)
#
#     parser.add_argument('--loss_scale', default=128, type=int)
#     parser.add_argument('--clip_value', default=4, type=int)
#
#     args, unknow_args = parser.parse_known_args()
#     for arg in unknow_args:
#         arg = arg.replace('--','')
#         k, v = arg.split('=')
#         if k not in args.__dict__:
#             args.__dict__[k] = v
#     return args