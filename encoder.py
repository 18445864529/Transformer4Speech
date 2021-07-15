import torch.nn as nn
from embedding import VGGExtractor
from position_encoding import PositionEncoding
from attention import MultiHeadAttention
from feed_forward import FeedForward
from mask import get_attn_pad_mask


class EncoderLayer(nn.Module):

    def __init__(self, ahead, adim, ffdim, dropout_rate):
        super().__init__()
        self.attention = MultiHeadAttention(ahead, adim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(adim)
        self.feedforward = FeedForward(adim, ffdim, dropout_rate)

    def forward(self, x, mask):
        res = x
        out = self.attention(x, x, x, mask)
        x = self.norm(res + self.dropout(out))
        out = self.feedforward(x)
        x = self.norm(res + self.dropout(out))
        return x


class Encoder(nn.Module):
    def __init__(self, idim, nlayer, ahead, adim, ffdim, dropout_rate):
        super().__init__()
        self.embed = VGGExtractor(idim, adim)
        self.position_encode = PositionEncoding(adim)
        self.layers = nn.ModuleList()
        for i in range(nlayer):
            self.layers.append(EncoderLayer(ahead, adim, ffdim, dropout_rate))

    def forward(self, x, x_len):
        # x is already padded by collate_fn in dataloader, shape: B, T, F.
        assert x_len.max() == x.size(1)
        x, x_len = self.embed(x, x_len)
        x += self.position_encode(x)
        mask = get_attn_pad_mask(x, x_len)
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
        return x, mask
