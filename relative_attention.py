import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, x, pos_emb, u, v, attn_mask=None, mems=None):
        x = x.permute(1, 0, 2)
        qlen, rlen, bsz = x.size(0), pos_emb.size(0), x.size(1) #rlen=klen
        if mems is not None and mems.nelement() != 0:
            mems = mems.permute(1, 0, 2) #T, B, d_model
            cat = torch.cat([mems, x], 0)

            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            R = self.r_net(pos_emb)

            Q, K, V = torch.chunk(w_heads, 3, dim=-1)
            Q = Q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(x))
            else:
                w_heads = self.qkv_net(x)
            R = self.r_net(pos_emb)        # r : L, B, d_model → L, B, d_model * n_head
            Q, K, V = torch.chunk(w_heads, 3, dim=-1)


        klen = K.size(0)

        Q = Q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        K = K.view(klen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head
        V = V.view(klen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head

        R = R.view(rlen, bsz, self.n_head, self.d_head)                # klen x n_head x d_head
        #print('qkr',Q.shape,K.shape, R.shape)
        #### compute attention score
        #u and v both (head * d_model)  r_w is u and r_r is v
        Qu = Q + u  # (L, B, h, d) + (h, d) = qlen x bsz x n_head x d_head
        #print('rwq and wk',rq.shape, k.shape)
        AC = torch.einsum('ibnd,jbnd->ijbn', (Qu, K))             # qlen x klen x bsz x n_head
        #AC = torch.einsum('ibnd,jbnd->ijbn', (Q, K))             # qlen x klen x bsz x n_head
        Qv = Q + v
        #print('rrq and rk',rr_head_q.shape, r_head_k.shape)
        BD = torch.einsum('ibnd,jbnd->ijbn', (Qv, R))              # qlen x klen x bsz x n_head (200,200,8,4)
        #BD = torch.einsum('ibnd,jbnd->ijbn', (Q, R))              # qlen x klen x bsz x n_head (200,200,8,4)
        BD = self._rel_shift(BD)
        #print('qlen:',qlen,'klen:',klen)
        #print('AC:',AC.shape,'BD:',BD.shape)
        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        #print(attn_score.shape)
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).eq(0)  # (batch, 1, 1(q), time)
            attn_mask = attn_mask.permute(2,3,0,1).repeat(1,1,1,self.n_head).to(torch.uint8) \
                if float(torch.__version__[:3]) <= 1.2 else attn_mask.permute(2,3,0,1).repeat(1,1,1,self.n_head).bool()
            #print('att_mask:',attn_mask.shape,'att_score:', attn_score.shape) #T, T, B, H
            min_value = float(numpy.finfo(torch.tensor(0, dtype=attn_score.dtype).numpy().dtype).min)
            #print('att_score',attn_score.shape)
            attn_score = attn_score.float().masked_fill(
                    attn_mask, min_value).type_as(attn_score)  #TTBH,1TBH
            #print('att_mask',attn_mask.permute(2,3,0,1)[0])
            #print('att_score:', attn_score.permute(2,3,0,1)[0])

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, V))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = x + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(x + attn_out)
        output=output.permute(1,0,2)
        #print(output.permute(0,2,1)[0][0][:])
        return output   #batch, x_len, d_model
