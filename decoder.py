import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward
from mask import get_attn_subsequent_mask

class DecoderLayer(nn.Module):

    def __init__(self, ahead, adim, ffdim, dropout_rate):
        super().__init__()
        self.attention = MultiHeadAttention(ahead, adim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(adim)
        self.feedforward = FeedForward(adim, ffdim, dropout_rate)

    def forward(self, y, subseq_mask, h, h_mask):
        res = y
        # self attention
        out = self.attention(y, y, y, subseq_mask)
        # add&norm
        y = self.norm(res + self.dropout(out))

        res = y
        # source attention
        out = self.attention(y, h, h, h_mask)
        # add&norm
        y = self.norm(res + self.dropout(out))

        res = y
        # feed forward part
        out = self.feedforward(y)
        # add&norm
        y = self.norm(res + self.dropout(out))
        return y


class Decoder(nn.Module):
    def __init__(self, odim, nlayer, ahead, adim, ffdim, dropout_rate):
        super().__init__()
        self.emb = nn.Embedding(odim, adim, padding_idx=0)
        self.layers = nn.ModuleList()
        for i in range(nlayer):
            self.layers.append(DecoderLayer(ahead, adim, ffdim, dropout_rate))
        self.final_fc = nn.Linear(adim, odim)

    def forward(self, y, h, h_mask):
        subseq_mask = get_attn_subsequent_mask(y)
        y = self.emb(y)
        for i, layer in enumerate(self.layers):
            y = layer(y, subseq_mask, h, h_mask)
        y = self.final_fc(y)
        return y
