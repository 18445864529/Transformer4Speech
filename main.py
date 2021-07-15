import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadedAttention as MH
from feed_forward import PositionwiseFeedForward as FF


class FF(nn.Module):

    def __init__(self, idim, hidden_units, dropout_rate):
        super().__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class MH(nn.Module):

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)


class EncoderLayer(nn.Module):

    def __init__(self, MH, FF, adim, ffdim, dropout_rate):
        super().__init__()
        self.att = MH(1, adim, dropout_rate)
        self.drop = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(adim)
        self.ff = FF(adim, ffdim, dropout_rate)

    def forward(self, x):
        res = x
        # attention part
        out = self.att(x, x, x, mask=None)
        # add&norm
        x = self.norm(res + self.drop(out))
        # feed forward part
        out = self.ff(x)
        # add&norm
        x = self.norm(res + self.drop(out))
        return x

class DecoderLayer(nn.Module):

    def __init__(self, MH, FF, adim, ffdim, dropout_rate):
        super().__init__()
        self.att = MH(1, adim, dropout_rate)
        self.drop = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(adim)
        self.ff = FF(adim, ffdim, dropout_rate)

    def forward(self, x):
        res = x
        # attention part
        out = self.att(x, x, x, mask=None)
        # add&norm
        x = self.norm(res + self.drop(out))
        # feed forward part
        out = self.ff(x)
        # add&norm
        x = self.norm(res + self.drop(out))
        return x

class Encoder(nn.Module):
    def __init__(self, nlayer, MH, FF, adim, ffdim, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(nlayer):
            self.layers.append(EncoderLayer(MH, FF, adim, ffdim, dropout_rate))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, nlayer, MH, FF, adim, ffdim, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(nlayer):
            self.layers.append(DecoderLayer(MH, FF, adim, ffdim, dropout_rate))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

class Transformer(nn.Module):
    def __init__(self, idim, en_layer, de_layer, adim, ffdim, MH, FF, dropout_rate):
        super().__init__()
        self.embedding = nn.Linear(idim, adim)
        self.encoder = Encoder(en_layer, MH, FF, adim, ffdim, dropout_rate)
        self.decoder = Decoder(de_layer, MH, FF, adim, ffdim, dropout_rate)
        self.fc = nn.Linear(adim, idim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.fc(x)
        return x



# model = Encoder(8, MH, FF, adim=256, ffdim=1024, dropout_rate=0.3).cuda()
# x = torch.randint(0, 10, size=(80, 300, 256)).float()
# y = torch.randint(11, 20, size=(80, 300, 256)).float()
# optim = torch.optim.Adam(model.parameters(), 0.01)
# loss_fn = nn.MSELoss(reduction='sum')
# start = time.time()
# for i in range(2):
#     x = x.cuda()
#     y = y.cuda()
#     out = model(x)
#     loss = loss_fn(out, y)
#     print(i, loss.item())
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
# end = time.time()
# print(end-start)


class CAE(nn.Module):
    """
    Thanks to http://dl-kento.hatenablog.com/entry/2018/02/22/200811#Convolutional-AutoEncoder
    """

    def __init__(self, z_dim=40):
        super().__init__()

        # define the network
        # encoder
        self.conv1 = nn.Sequential(nn.ZeroPad2d((1,2,2,2)),
                                   nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=0), #32*128 to 32*64
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1,2,2,2)),
                                   nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0), #64*64 to 32*32
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(), nn.Dropout(0.2))
        self.conv3 = nn.Sequential(nn.ZeroPad2d((1,0,1,0)),
                                   nn.Conv2d(64, 128, kernel_size=3, stride=2), #16*16
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(), nn.Dropout(0.3))
        self.conv4 = nn.Sequential(nn.ZeroPad2d((1,0,1,0)),
                                   nn.Conv2d(128, 256, kernel_size=3, stride=2), #8*8
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(), nn.Dropout(0.3))
        self.conv5 = nn.Sequential(nn.ZeroPad2d((1,0,1,0)),
                                   nn.Conv2d(256, 512, kernel_size=3, stride=2), #8*8
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(), nn.Dropout(0.4))
        self.fc1 = nn.Conv2d(512, z_dim, kernel_size=2)

        # decoder
        self.fc2 = nn.Sequential(nn.ConvTranspose2d(z_dim, 512, kernel_size=2),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(), nn.Dropout(0.4))
        self.conv5d = nn.Sequential(
                                    nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(), nn.Dropout(0.3))
        self.conv4d = nn.Sequential(
                                    nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(), nn.Dropout(0.3))
        self.conv3d = nn.Sequential(
                                    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(), nn.Dropout(0.2))
        self.conv2d = nn.Sequential(
                                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
        self.conv1d = nn.Sequential(
                                    nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1))

    def forward(self, x):
        # encoded = self.fc1(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        # print('before conv1', x.shape)
        encoded = self.conv1(x)
        # print('before conv2', encoded.shape)
        encoded = self.conv2(encoded)
        # print('before conv3', encoded.shape)
        encoded = self.conv3(encoded)
        # print('before conv4', encoded.shape)
        encoded = self.conv4(encoded)
        # print('before conv5',encoded.shape)
        encoded = self.conv5(encoded)
        # print('before fc1',encoded.shape)
        encoded = self.fc1(encoded)
        # print('before fc2',encoded.shape)
        decoded = self.fc2(encoded)
        # print('before conv5',decoded.shape)
        decoded = self.conv5d(decoded)
        # print('before conv4',decoded.shape)
        decoded = self.conv4d(decoded)
        # print('before conv3',decoded.shape)
        decoded = self.conv3d(decoded)
        # print('before conv2',decoded.shape)
        decoded = self.conv2d(decoded)
        # print('before conv1',decoded.shape)
        decoded = self.conv1d(decoded)
        # print('before sigmoid',decoded.shape)
        decoded = nn.Sigmoid()(decoded)
        # exit()
        return decoded