
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import L1Loss
from scipy.sparse import diags
from data import BitcoinDataset, DataIterator
from util import Profiler

def infer(args, model, val_iterator=None):
    training = val_iterator is not None
    if not training:
        model = model.cuda()
        val_dataset = BitcoinDataset(
            args.price_path, args.tweet_path, date_from=args.start_date, date_to=args.end_date)
        val_iterator = DataIterator(val_dataset, args.batch_size, val=True)
    criterion = L1Loss(reduction='none')
    count = len(val_iterator)

    print('Running inference on {} datapoints...'.format(count))
    profiler = Profiler(['infer', 'fw'])
    results, losses, losses_daily = [], [], []
    model.eval()
    for i, (price, tweet, trgt) in enumerate(val_iterator):
        profiler.start('fw')
        with torch.no_grad():
            out = model(price, tweet)
            loss = criterion(out, trgt)
        results.append([out, loss])
        loss_daily = loss.mean(axis=0).view(7, 24).mean(axis=1)
        loss = loss.mean()
        losses_daily.append(loss_daily)
        losses.append(loss)
        profiler.stop('fw')
        profiler.bump('infer')

        if not training and (profiler.totals['infer'] > 60 or i == count // args.batch_size):
            avg_loss = torch.stack(losses).mean().item()
            avg_loss_daily = torch.stack(losses_daily).mean(axis=0).tolist()

            print(' | '.join([
                f'[{min((i+1) * args.batch_size, count):{len(str(count))}}/{count}] loss: {avg_loss:.4f}',
                ('loss-daily: [' + ', '.join(['{:.4f}']*7) + ']').format(*avg_loss_daily),
                f'{profiler.means["infer"]:.3f}s/{args.batch_size}-batch' + \
                f'(fw: {profiler.means["fw"]:.3f}s, bw: {profiler.means["bw"]:.3f}s)',
            ]), flush=True)

            profiler.reset()
    
    results = [torch.cat(r, dim=0).cpu() for r in zip(*results)]
    if not training:
        take = 2
        mean, std = 7.9078, 1.5308
        out = (results[0].numpy() * std) + mean
        out = diags(out.T[:take], offsets=np.arange(take), shape=(out.shape[0], out.shape[0]+take))
        out = np.asarray(out.sum(axis=0)).T[take-1:-take-1] / take
        trgt = (val_dataset.price_trgt.numpy()[take//2:-take//2, 0] * std) + mean
        date_from = pd.Timestamp('2020-09-01') + pd.Timedelta(days=30, hours=take//2)
        date_to = pd.Timestamp('2021-02-01') - pd.Timedelta(days=7, hours=take//2+1)
        date_range = pd.date_range(date_from, date_to, freq='H')
        pd.DataFrame(
            data=np.concatenate([trgt, out], axis=-1),
            index=date_range,
            columns=['actual', 'forecast']
        ).to_csv(args.output, index_label='timestamp')
    loss = results[1].mean().item()
    loss_daily = results[1].mean(axis=0).view(7, 24).mean(axis=1).tolist()
    print(' | '.join([
        f'[Inference] loss: {loss:.4f}',
        ('loss-daily: [' + ', '.join(['{:.4f}']*7) + ']').format(*loss_daily)
    ]), flush=True)
    return loss, loss_daily
