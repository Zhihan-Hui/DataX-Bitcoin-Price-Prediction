
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings


class FourierEncoding(nn.Module):
    def __init__(self, max_freq, num_bands):
        super(FourierEncoding, self).__init__()
        self.register_buffer('freqs', torch.linspace(0, math.log(max_freq / 2), num_bands).exp() * math.pi)

    def forward(self, x):
        x_d = torch.linspace(-1, 1, x.size(0), device=x.device).unsqueeze(-1).unsqueeze(-1)
        pos = x_d * self.freqs.unsqueeze(0)
        pos = torch.cat([pos.sin(), pos.cos(), x_d], dim=-1)
        x = torch.cat([x, pos.expand(-1, x.size(1), -1)], dim=-1)
        return x


class LowRankLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, rank=None):
        super(LowRankLinear, self).__init__(in_features, out_features, bias)
        self.rank = rank
        if rank is not None:
            U, S, V = torch.pca_lowrank(self.weight.detach(), q=rank, center=False)
            U, V = U @ S.sqrt().diag(), V @ S.sqrt().diag()
            self.weight_u  = nn.Parameter(U)
            self.weight_v = nn.Parameter(V.T)
            self.weight = None

    def forward(self, x):
        return F.linear(x, self.weight if self.rank is None else self.weight_u @ self.weight_v, self.bias)


class ReLUDropout(nn.Dropout):
    # adapted from https://gist.github.com/vadimkantorov/360ece06de4fd2641fa9ed1085f76d48
    def forward(self, x):
        if not self.training or self.p == 0:
            return x.clamp_(min=0) if self.inplace else x.clamp(min=0)
        mask = (torch.rand_like(x) < self.p) | (x < 0)
        return x.masked_fill_(mask, 0).mul_(1 / (1 - self.p)) if self.inplace \
            else x.masked_fill(mask, 0).mul(1 / (1 - self.p))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=768, nhead=12, dim_ff=3072, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.reludrop = ReLUDropout(dropout, inplace=True)

    def forward_prenorm(self, src):
        x = self.norm1(src)
        src = src + self.dropout(self.attn(x, x, x, need_weights=False)[0])
        x = self.norm2(src)
        src = src + self.dropout(self.linear2(self.reludrop(self.linear1(x))))
        return src

    def forward_postnorm(self, src):
        src = self.norm1(src + self.dropout(self.attn(src, src, src, need_weights=False)[0]))
        src = self.norm2(src + self.dropout(self.linear2(self.reludrop(self.linear1(src)))))
        return src

    def forward(self, src):
        return self.forward_prenorm(src)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=768, nhead=12, dim_ff=3072, dropout=0.1, input_dim=None):
        super(TransformerDecoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, kdim=input_dim, vdim=input_dim)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.reludrop = ReLUDropout(dropout, inplace=True)

    def forward_prenorm(self, trgt, memory):
        x = self.norm1(trgt)
        trgt = trgt + self.dropout(self.attn(x, memory, memory, need_weights=False)[0])
        x = self.norm2(trgt)
        trgt = trgt + self.dropout(self.linear2(self.reludrop(self.linear1(x))))
        return trgt

    def forward_postnorm(self, trgt, memory):
        trgt = self.norm1(trgt + self.dropout(self.attn(trgt, memory, memory, need_weights=False)[0]))
        trgt = self.norm2(trgt + self.dropout(self.linear2(self.reludrop(self.linear1(trgt)))))
        return trgt

    def forward(self, trgt, memory):
        return self.forward_prenorm(trgt, memory)


class Perceiver(nn.Module):
    def __init__(self, out_len, d_model=768, d_input=768, dropout=0.1,
                       num_cross_attn=8, enc_dec_ratio=6, max_freqs=[1], band_size=31):
        super(Perceiver, self).__init__()
        self.query = nn.Parameter(torch.randn(out_len, d_model))
        self.pos = nn.ModuleList([FourierEncoding(freq, band_size//2) for freq in max_freqs])
        # self.init_layers = nn.ModuleList([
        #     TransformerDecoderLayer(d_model, 1, d_model, dropout, input_dim=d_input+band_size),
        #     *[TransformerEncoderLayer(d_model, d_model//band_size, d_model, dropout) for _ in range(enc_dec_ratio)]
        # ])
        # self.subseq_layers = nn.ModuleList([
        #     TransformerDecoderLayer(d_model, 1, d_model, dropout, input_dim=d_input+band_size),
        #     *[TransformerEncoderLayer(d_model, d_model//band_size, d_model, dropout) for _ in range(enc_dec_ratio)]
        # ])
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, 1, d_model, dropout, input_dim=d_input+band_size),
            *[TransformerEncoderLayer(d_model, d_model//band_size, d_model, dropout) for _ in range(enc_dec_ratio)]
        ])
        self.norm = nn.LayerNorm(d_input+band_size)
        self.num_cross_attn = num_cross_attn

    def forward(self, *xs):
        x = self.norm(torch.cat([pos_i(x_i) for pos_i, x_i in zip(self.pos, xs)]))
        q = self.query.unsqueeze(1).expand(-1, x.size(1), -1)
        for i in range(self.num_cross_attn):
            q = self.layers[0](q, x)
            for layer in self.layers[1:]:
                q = layer(q)
        return q


class Model(nn.Module):
    def __init__(self, path, rnn_dim, rnn_layers, rnn_dropout, perceiver_dim, perceiver_dropout,
                       num_cross_attn, enc_dec_ratio, max_freqs, band_size, infer=False):
        super(Model, self).__init__()
        self.path = path
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        self.rnn_dropout = rnn_dropout
        self.perceiver_dim = perceiver_dim
        self.perceiver_dropout = perceiver_dropout
        self.num_cross_attn = num_cross_attn
        self.enc_dec_ratio = enc_dec_ratio
        self.max_freqs = max_freqs
        self.band_size = band_size
        self.infer = infer

        self.price_hidden = nn.Parameter(torch.randn(rnn_layers, rnn_dim))
        self.price_encoder = nn.GRU(4, rnn_dim, rnn_layers, dropout=rnn_dropout)
        self.tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
        roberta = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
        self.sentiment_encoder = roberta.roberta
        self.sentiment_classifier = roberta.classifier
        self.sentiment_dropout = nn.Dropout(perceiver_dropout, inplace=True)
        self.sentiment_projection = LowRankLinear(3+768, rnn_dim, rank=rnn_dim//4)
        self.decoder = Perceiver(168, d_model=perceiver_dim, d_input=rnn_dim, dropout=perceiver_dropout,
                                 num_cross_attn=num_cross_attn, enc_dec_ratio=enc_dec_ratio,
                                 max_freqs=max_freqs, band_size=band_size)
        self.projection = nn.Linear(perceiver_dim, 1)
        self.initialize()

    def initialize(self):
        for param in self.sentiment_encoder.parameters():
            param.requires_grad = False
        for param in self.sentiment_classifier.parameters():
            param.requires_grad = False

    def forward_base(self, price, sentiment):
        """
        Shapes (N=batch, L=seq_len, E=embed_dim):
        - price: (L=720, N, E=4)
        - sentiment: (L=224, N, E=3+768)
        - out: (N, L=168, E=1)
        """
        price_hidden = self.price_hidden.unsqueeze(1).repeat(1, price.size(1), 1)
        price_out = self.price_encoder(price.contiguous(), price_hidden)[0]
        sentiment_out = self.sentiment_projection(self.sentiment_dropout(sentiment))
        out = self.projection(self.decoder(price_out, sentiment_out)).transpose(0, 1)
        return out

    def forward_infer(self, price, text):
        """
        Shapes (N=batch, L=seq_len, E=embed_dim):
        - price: (L=720, N, E=4)
        - text: list of len N*224
        """
        device = next(self.parameters()).device
        price = price.to(device)
        token = self.tokenizer(text, padding=True, max_length=512, truncation=True, return_tensors='pt')
        input_ids, attention_mask = token['input_ids'].to(device), token['attention_mask'].to(device)
        input_ids, attention_mask = input_ids.split(32), attention_mask.split(32)
        sentiments = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for iids, mask in zip(input_ids, attention_mask):
                embedding = self.sentiment_encoder(iids, mask)
                embedding = embedding['last_hidden_state']
                sentiment = self.sentiment_classifier(embedding).softmax(dim=-1)
                sentiments.append(torch.cat([sentiment, embedding[:, 0]], dim=-1)) # 'CLS' token
            sentiments = torch.cat(sentiments).view(-1, 32*7, 3+768).transpose(0, 1)
        return self.forward_base(price, sentiments)

    def forward(self, price, tweet):
        return (self.forward_infer if self.infer else self.forward_base)(price, tweet)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path)
        rnn_dim = ckpt.pop('rnn_dim')
        rnn_layers = ckpt.pop('rnn_layers')
        rnn_dropout = ckpt.pop('rnn_dropout')
        perceiver_dim = ckpt.pop('perceiver_dim')
        perceiver_dropout = ckpt.pop('perceiver_dropout')
        num_cross_attn = ckpt.pop('num_cross_attn')
        enc_dec_ratio = ckpt.pop('enc_dec_ratio')
        max_freqs = ckpt.pop('max_freqs')
        band_size = ckpt.pop('band_size')
        model = cls(path, rnn_dim, rnn_layers, rnn_dropout, perceiver_dim, perceiver_dropout,
                    num_cross_attn, enc_dec_ratio, max_freqs, band_size)
        model.load_state_dict(ckpt.pop('model'))
        return model, ckpt

    def save(self, optimizer, scheduler, iteration, val_loss=None):
        # suppress UserWarning: save or load the state of the optimzer
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            ckpt = {
                'rnn_dim': self.rnn_dim,
                'rnn_layers': self.rnn_layers,
                'rnn_dropout': self.rnn_dropout,
                'perceiver_dim': self.perceiver_dim,
                'perceiver_dropout': self.perceiver_dropout,
                'num_cross_attn': self.num_cross_attn,
                'enc_dec_ratio': self.enc_dec_ratio,
                'max_freqs': self.max_freqs,
                'band_size': self.band_size,
                'model': self.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iteration': iteration,
            }
            if val_loss is not None:
                ckpt['val_loss'] = val_loss
            torch.save(ckpt, self.path)



if __name__ == '__main__':
    from data import BitcoinDataset, WeightedSampler
    from torch.utils.data import DataLoader
    from prettytable import PrettyTable

    def count_parameters(model):
        table = PrettyTable(['Modules', 'Parameters'])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f'Total Trainable Params: {total_params}')
        return total_params

    model = Model(path='./model.pth', rnn_dim=32, rnn_layers=1, rnn_dropout=0,
                  perceiver_dim=32, perceiver_dropout=0.3, num_cross_attn=8,
                  enc_dec_ratio=1, max_freqs=[60, 14], band_size=7, infer=False)
    # print(f'{sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters')
    count_parameters(model)

    dataset = BitcoinDataset('./btc-price.csv', './btc-tweet.npy', date_to='2021-01-01')
    sampler = WeightedSampler(dataset.price_inpt.numpy(), dataset.price_trgt.numpy(), alpha=0)
    loader = DataLoader(
        dataset, batch_size=2, sampler=sampler, num_workers=2, collate_fn=dataset.collate)
    for price, tweet, trgt in loader:
        print(model(price, tweet).shape, trgt.shape); break
