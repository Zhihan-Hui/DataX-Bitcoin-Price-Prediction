
import os
import argparse
import torch
from model import Model
from train import train
from infer import infer

def parse():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    parser_train = subparsers.add_parser('train')
    # Model params
    parser_train.add_argument('--model-path', type=str, required=True)
    parser_train.add_argument('--rnn-dim', type=int, default=32)
    parser_train.add_argument('--rnn-layers', type=int, default=1)
    parser_train.add_argument('--rnn-dropout', type=float, default=0)
    parser_train.add_argument('--perceiver-dim', type=int, default=32)
    parser_train.add_argument('--perceiver-dropout', type=float, default=0.3)
    parser_train.add_argument('--num-cross-attn', type=int, default=24)
    parser_train.add_argument('--enc-dec-ratio', type=int, default=1)
    parser_train.add_argument('--max-freqs', type=int, nargs=2, default=[60, 14])
    parser_train.add_argument('--band-size', type=int, default=7)
    parser_train.add_argument('--finetune', type=str, default=None)
    # Data params
    parser_train.add_argument('--price-path', type=str, required=True)
    parser_train.add_argument('--tweet-path', type=str, required=True)
    parser_train.add_argument('--start-date', type=str, default=None)
    parser_train.add_argument('--val-date', type=str, default=None)
    parser_train.add_argument('--end-date', type=str, default=None)
    parser_train.add_argument('--batch-size', type=int, default=32)
    parser_train.add_argument('--sampler-alpha', type=float, default=0.1)
    # Training params
    parser_train.add_argument('--iters', type=int, default=72000)
    parser_train.add_argument('--lr', type=float, default=1e-5)
    parser_train.add_argument('--warmup', type=int, default=720)
    parser_train.add_argument('--warmup-init', type=float, default=0)
    parser_train.add_argument('--min-lr-ratio', type=float, default=0.1)
    parser_train.add_argument('--weight-decay', type=float, default=1e-3)
    parser_train.add_argument('--val-freq', type=int, default=1000)
    parser_train.add_argument('--verbose-ckpt', action='store_true')
    parser_train.add_argument('--logdir', type=str, default=None)

    parser_infer = subparsers.add_parser('infer')
    parser_infer.add_argument('--model-path', type=str, required=True)
    parser_infer.add_argument('--price-path', type=str, required=True)
    parser_infer.add_argument('--tweet-path', type=str, required=True)
    parser_infer.add_argument('--output', type=str, required=True)
    parser_infer.add_argument('--start-date', type=str, default=None)
    parser_infer.add_argument('--end-date', type=str, default=None)
    parser_infer.add_argument('--batch-size', type=int, default=32)

    return parser.parse_args()

def load_model(args):
    state = {'optimizer': None, 'scheduler': None, 'iteration': 0}
    if os.path.exists(args.model_path):
        model, state = Model.load(args.model_path)
        print(f'Model loaded from {args.model_path}')
    elif args.finetune:
        assert os.path.exists(args.finetune), 'Finetune path does not exist!'
        assert not os.path.exists(args.model_path), 'Model path exists already!'
        model, _ = Model.load(args.finetune)
        model.path = args.model_path
        print(f'Finetuning model from {args.finetune} to {args.model_path}')
    elif args.command == 'train':
        model = Model(args.model_path, args.rnn_dim, args.rnn_layers, args.rnn_dropout,
                      args.perceiver_dim, args.perceiver_dropout, args.num_cross_attn,
                      args.enc_dec_ratio, args.max_freqs, args.band_size)
        print(f'Model created for {args.model_path}')
    else:
        raise RuntimeError('Failure in loading models.')
    return model, state

def main():
    args = parse()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    model, state = load_model(args)
    if args.command == 'train':
        assert args.val_date is not None or args.val_freq == 0, 'Specify date for validation or set val-freq to 0'
        assert args.band_size % 2, 'Band-size must be 2*k+1 where k is the number of bands'
        train(args, model, state)
    else:
        infer(args, model)

if __name__ == '__main__':
    main()
