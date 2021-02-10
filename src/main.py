import argparse
import logging
import torch
from run_config import RunConfig
from data_process import xx
torch.cuda.set_device(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='book')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--raa_dim', type=int, default=16)
    parser.add_argument('--att1_dim', type=int, default=4)
    parser.add_argument('--att2_dim', type=int, default=4)
    parser.add_argument('--lamda', type=int, default=0.1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0.00001)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--hop', type=int, default=2)
    parser.add_argument('--num_u2t', type=int, default=18)
    parser.add_argument('--num_i2t', type=int, default=12)

    return parser.parse_args()


if __name__ == '__main__':
    logging.info('=' * 30)
    logging.info('BEGIN UPIACM')
    logging.info('=' * 30)
    xx()

    args = get_args()
    run_config = RunConfig(args=args)
    run_config.train()
