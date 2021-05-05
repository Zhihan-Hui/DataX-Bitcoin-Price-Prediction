
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import L1Loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from collections import OrderedDict
from datetime import date, timedelta
from glob import glob
from data import BitcoinDataset, DataIterator
from infer import infer
from util import Profiler


def train(args, model, state):
    def cosine_scheduler(optimizer, warmup_steps, training_steps, warmup_init=0, min_lr_ratio=0):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return (1 - warmup_init) / warmup_steps * current_step + warmup_init
            progress = (current_step - warmup_steps) / (training_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + np.cos(np.pi * progress))
        return LambdaLR(optimizer, lr_lambda)

    model = model.cuda()
    optimizer = AdamW([param for param in model.parameters() if param.requires_grad], args.lr)
    scheduler = cosine_scheduler(optimizer, args.warmup, args.iters, args.warmup_init, args.min_lr_ratio)
    if state['optimizer'] is not None:
        optimizer.load_state_dict(state['optimizer'])
    if state['scheduler'] is not None:
        scheduler.load_state_dict(state['scheduler'])
    iteration = state['iteration']
    criterion = L1Loss(reduction='none')
    scaler = torch.cuda.amp.GradScaler()

    train_dataset = BitcoinDataset(
        args.price_path, args.tweet_path, date_from=args.start_date, date_to=args.val_date)
    data_iterator = DataIterator(train_dataset, args.batch_size, alpha=args.sampler_alpha, val=False)
    if args.val_freq:
        val_dataset = BitcoinDataset(
            args.price_path, args.tweet_path,
            date_from=date.fromisoformat(args.val_date)-timedelta(days=30), date_to=args.end_date)
        val_iterator = DataIterator(val_dataset, args.batch_size, val=True)

    if args.logdir is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.logdir)
    
    ckpts = glob(os.path.splitext(model.path)[0]+'-*')
    ckpt_num = 0 if len(ckpts) == 0 else int(max(ckpts)[-6:-4])+1
    min_loss = 1 if ckpt_num == 0 else state.pop('val_loss')
    
    print('Training model for {} iterations...'.format(args.iters))
    profiler = Profiler(['train', 'fw', 'bw'])
    model.train()
    while iteration < args.iters:
        losses, losses_daily = [], []
        for price, tweet, trgt in data_iterator:
            if iteration >= args.iters:
                break

            profiler.start('fw')
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = criterion(model(price, tweet), trgt)
            loss_daily = loss.mean(axis=0).view(7, 24).mean(axis=1)
            loss = loss.mean()
            losses_daily.append(loss_daily)
            losses.append(loss)
            profiler.stop('fw')

            profiler.start('bw')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            profiler.stop('bw')

            del price, tweet, trgt

            iteration += 1
            profiler.bump('train')
            if profiler.totals['train'] > 60 or iteration == args.iters:
                avg_loss = torch.stack(losses).mean().item()
                avg_loss_daily = torch.stack(losses_daily).mean(axis=0).tolist()
                lr = optimizer.param_groups[0]['lr']

                print(' | '.join([
                    f'[{iteration:{len(str(args.iters))}}/{args.iters}] loss: {avg_loss:.4f}',
                    ('loss-daily: [' + ', '.join(['{:.4f}']*7) + ']').format(*avg_loss_daily),
                    f'{profiler.means["train"]:.3f}s/{args.batch_size}-batch' + \
                    f'(fw: {profiler.means["fw"]:.3f}s, bw: {profiler.means["bw"]:.3f}s)',
                    f'lr: {lr:.2g}',
                ]), flush=True)

                if args.logdir is not None:
                    writer.add_scalar('loss', avg_loss, iteration)
                    writer.add_scalars('loss-daily', {f'day{i+1}': l for i, l in enumerate(avg_loss_daily)}, iteration)
                    writer.add_scalar('lr', lr, iteration)

                model.save(optimizer, scheduler, iteration)

                profiler.reset()
                del losses[:], losses_daily[:]

            if iteration == args.iters or (args.val_freq > 0 and iteration % args.val_freq == 0):
                loss, loss_daily = infer(args, model, val_iterator)
                if args.logdir is not None:
                    writer.add_scalar('val-loss', loss, iteration)
                    writer.add_scalars('val-loss-daily', {f'day{i+1}': l for i, l in enumerate(loss_daily)}, iteration)
                if args.verbose_ckpt and loss < min_loss:
                    path = model.path
                    fname, ext = os.path.splitext(path)
                    model.path = f'{fname}-{ckpt_num:02d}{ext}'
                    model.save(optimizer, scheduler, iteration)
                    print(f'Saved model to {model.path}')
                    model.path = path
                    min_loss, ckpt_num = loss, ckpt_num+1
                del loss, loss_daily
                model.train()

    if args.logdir is not None:
        writer.close()
