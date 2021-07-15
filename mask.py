"""
Modified from
https://github.com/sooftware/speech-transformer/
"""

import torch


def get_attn_pad_mask(x, x_len):
    """ mask position is set to True(1) """

    def get_transformer_non_pad_mask(x, x_len):
        """ Padding position is set to 0, either use x_len or pad_id """
        batch_size = x.size(0)

        if len(x.size()) == 2:
            non_pad_mask = x.new_ones(x.size())  # B x T
        elif len(x.size()) == 3:
            non_pad_mask = x.new_ones(x.size()[:-1])  # B x T
        else:
            raise ValueError(f"Unsupported input shape {x.size()}")

        for i in range(batch_size):
            non_pad_mask[i, x_len[i]:] = 0

        return non_pad_mask

    non_pad_mask = get_transformer_non_pad_mask(x, x_len)
    pad_mask = non_pad_mask.lt(1)
    attn_pad_mask = pad_mask.unsqueeze(1)  # .expand(-1, x.size(1), -1)
    return attn_pad_mask


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)

    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask.eq(1)


def add_sos_eos(y, padded_value=0, sos_eos_value=-1):
    sos = y.new_full((y.size(0), 1), sos_eos_value)
    sos_y = torch.cat([sos, y], dim=-1)
    eos = y.new([sos_eos_value])
    y_body = [line[line != padded_value] for line in y]
    y_pad = [line[line == padded_value] for line in y]
    y_eos = []
    for yb, yp in zip(y_body, y_pad):
        y_eos.append(torch.cat([yb, eos, yp], dim=-1))
    y_eos = torch.stack(y_eos)
    return sos_y, y_eos


if __name__ == '__main__':
    inp = torch.rand(3, 7, 4)
    lens = torch.tensor([5, 7, 2])
    # print(get_attn_pad_mask(inp, lens))
    y = torch.tensor([[1, 2, 0, 0], [4, 0, 0, 0], [2, 4, 5, 5]])
    # print(get_attn_subsequent_mask(y))
    s, e = add_sos_eos(y)
    e = torch.nn.Embedding(6, 4, padding_idx=0)(y)
    print(e)
