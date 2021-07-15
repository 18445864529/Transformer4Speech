from encoder import Encoder
from decoder import Decoder
from mask import add_sos_eos
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, idim, odim, nlayer=6, ahead=4, adim=512, ffdim=1024, dropout_rate=0.3):
        super(Transformer, self).__init__()
        self.encoder = Encoder(idim, nlayer, ahead, adim, ffdim, dropout_rate)
        self.decoder = Decoder(odim, nlayer, ahead, adim, ffdim, dropout_rate)
        self.odim = odim
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, x_len, y):
        sos_y, y_eos = add_sos_eos(y, padded_value=0, sos_eos_value=self.odim-1)
        h, h_mask = self.encoder(x, x_len)
        out = self.decoder(sos_y, h, h_mask)
        loss = self.loss_fn(out.transpose(-2,-1).contiguous(), y_eos)
        return loss, out
        # return self.decoder(*((sos_y,) + self.encoder(x, x_len)))
