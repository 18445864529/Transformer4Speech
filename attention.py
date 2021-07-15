import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        assert n_feat % n_head == 0
        self.dim_head = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask):
        B = q.size(0)
        q = self.linear_q(q).view(B, -1, self.h, self.dim_head).transpose(1, 2)  # B, head, T_q, d_head
        k = self.linear_k(k).view(B, -1, self.h, self.dim_head).transpose(1, 2)  # B, head, T_k, d_head
        v = self.linear_v(v).view(B, -1, self.h, self.dim_head).transpose(1, 2)  # B, head, T_v, d_head

        score = torch.matmul(q, k.transpose(-2, -1).contiguous()) / math.sqrt(self.dim_head)  # (batch, head, T_q, T_k)
        if mask is not None:
            mask = mask.unsqueeze(1)  # B, 1, 1, T_k >> only the last dim matters! Others can broadcast.
            score = score.masked_fill(mask, -1e9)
        soft_score = torch.softmax(score, dim=-1)  # B, head, T_q, T_k
        # print(soft_score)
        att = torch.matmul(self.dropout(soft_score), v)  # B, head, T_q, d_head << because T_k == T_v
        att = att.transpose(1, 2).contiguous().view(B, -1, self.h * self.dim_head)  # B, T_q, adim << adim == d_head * head
        return self.linear_out(att)  # B, T_q, adim
