import argparse
import importlib
from pathlib import Path

import mindspore
from mindspore import context


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training params
    parser.add_argument('--algo', default='SenticBERT_ABSA', type=str, help='the algorithm need to train or eval.')
    parser.add_argument('--data_dir', default='/dataset/sentiment_analysis_data/SenticBERT_ABSA_data', type=str)
    parser.add_argument('--dataset', default='rest16', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--save_ckpt_path', default='/model/checkpoints/SenticBERT_ABSA/rest16/best_eval.ckpt', type=str)
    parser.add_argument('--valset_ratio', default=0.1, type=float)
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased', type=str, help='The tokenizer used in ABSA.')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--num_epochs', default=20, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--warmup', default=5e-2, type=float)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default='GPU', type=str, choices=['Ascend', 'CPU', 'GPU'])
    parser.add_argument('--seed', default=1000, type=int, help='set seed for reproducibility')
    # optimizer params
    parser.add_argument('--lr', default=5e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--weight_decay', default=0.00001, type=float)
    # train/eval
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'eval'])
    parser.add_argument('--graph_mode', default=True, type=bool)

    opt = parser.parse_args()
    mindspore.set_seed(opt.seed)
    if opt.graph_mode:
        context.set_context(mode=context.GRAPH_MODE, device_target=opt.device)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=opt.device)
    algo = importlib.import_module(opt.algo)
    print(f'Start running algorithm {opt.algo} in {opt.mode} mode.')
    if opt.mode == 'train':
        ins = algo.Instructor(opt)
        ins.train()
    else:
        ins = algo.Instructor(opt)
        ins.eval(opt.save_ckpt_path)

