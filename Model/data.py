
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, WeightedRandomSampler
from multiprocessing import cpu_count
from util import patchify, tweet_std


class BitcoinDataset(Dataset):
    def __init__(self, price_path, tweet_path, date_from=None, date_to=None):
        super(BitcoinDataset, self).__init__()
        price_hour = pd.read_csv(price_path, index_col='timestamp', parse_dates=True, infer_datetime_format=True, dtype=np.float32)
        idx_from = 0 if date_from is None else price_hour.index.get_loc(pd.Timestamp(date_from))
        idx_to = price_hour.shape[0] if date_to is None else price_hour.index.get_loc(pd.Timestamp(date_to))
        
        mean, std = 7.9078, 1.5308
        price_hour = ((np.log(price_hour.to_numpy()) - mean) / std).astype(np.float32)
        price_inpt = patchify(price_hour[idx_from:idx_to-168], 720)[360:-361, 0] # (N, 720, 4)
        price_trgt = patchify(price_hour[idx_from+720:idx_to, -1, np.newaxis], 168)[84:-85, 0] # (N, 168, 1)
        tweet = (np.load(tweet_path).reshape((-1, 40, 3+768)) / tweet_std()).reshape((-1, 40*(3+768)))
        tweet = patchify(tweet[idx_from//24+23:idx_to//24-7], 7)[3:-4].reshape((-1, 40*7, 3+768)) # (N/24, 280, 768)
        
        self.price_inpt = torch.from_numpy(price_inpt)
        self.price_trgt = torch.from_numpy(price_trgt)
        self.sentiments = torch.from_numpy(tweet)

    def __len__(self):
        return self.price_inpt.size(0)

    def __getitem__(self, idx):
        price_noise = torch.randn(720, 4) * 0.01
        sample = torch.multinomial(torch.ones(40*7), 32*7, replacement=False).sort().values
        return self.price_inpt[idx] + price_noise, self.sentiments[idx//24, sample], self.price_trgt[idx]

    def collate(self, batch):
        price, tweet, trgt = zip(*batch)
        return torch.stack(price, dim=1), torch.stack(tweet, dim=1), torch.stack(trgt)


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_output = next(self.loader)
        except StopIteration:
            self.next_output = None
            return
        with torch.cuda.stream(self.stream):
            self.next_output = [x.cuda(non_blocking=True) for x in self.next_output]

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        output = self.next_output
        if output is not None:
            for x in output:
                x.record_stream(torch.cuda.current_stream())
        self.preload()
        return output


class WeightedSampler(WeightedRandomSampler):
    def __init__(self, price_inpt, price_trgt, alpha=0):
        inpt_coef = self.linear_coef(price_inpt[..., 0])
        trgt_coef = self.linear_coef(price_trgt[..., 0])
        diff = np.abs(trgt_coef - inpt_coef)
        prob = alpha * (diff / diff.sum()) + np.full(diff.size, (1 - alpha) / diff.size)
        super(WeightedSampler, self).__init__(prob, prob.size, replacement=True)

    def linear_coef(self, data):
        x = np.arange(data.shape[1])
        x = x - x.mean()
        y = data - data.mean(axis=1, keepdims=True)
        return (x * y).mean(axis=1) / np.square(x).mean()


class DataIterator():
    def __init__(self, dataset, batch_size, alpha=0, val=False):
        if alpha:
            sampler = WeightedSampler(dataset.price_inpt.numpy(), dataset.price_trgt.numpy(), alpha)
            self.loader = DataLoader(
                dataset, batch_size=batch_size, sampler=sampler,
                num_workers=cpu_count(), pin_memory=True, collate_fn=dataset.collate)
        else:
            self.loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=not val,
                num_workers=cpu_count(), pin_memory=True, collate_fn=dataset.collate)
        self.prefetcher = DataPrefetcher(self.loader)
        self.val = val

    def __len__(self):
        return len(self.loader.dataset)
        
    def __iter__(self):
        yield from (self.val_iter if self.val else self.train_iter)()

    def train_iter(self):
        while True:
            output = self.prefetcher.next()
            if output is None:
                self.prefetcher = DataPrefetcher(self.loader)
                output = self.prefetcher.next()
            yield output

    def val_iter(self):
        output = self.prefetcher.next()
        while output is not None:
            yield output
            output = self.prefetcher.next()
        self.prefetcher = DataPrefetcher(self.loader)
