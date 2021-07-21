import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # self attention
        res = y
        out = self.attention(y, y, y, subseq_mask)
        y = self.norm(res + self.dropout(out))
        # source attention
        res = y
        out = self.attention(y, h, h, h_mask)
        y = self.norm(res + self.dropout(out))
        # feed forward
        res = y
        out = self.feedforward(y)
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
        """
        :param y: B, T
        :param h: encoder output
        :param h_mask: mask of encoder in/output
        :return: B, T, num_class
        """
        subseq_mask = get_attn_subsequent_mask(y)
        y = self.emb(y)
        for i, layer in enumerate(self.layers):
            y = layer(y, subseq_mask, h, h_mask)
        y = self.final_fc(y)
        return y

    def recognize_beam(self, encoder_outputs, mask, char_list, args):
        """Beam search from https://github.com/kaituoxu/Speech-Transformer/blob/master/src/transformer/decoder.py
        Args:
            encoder_outputs: T x H
            char_list: list of character
            args: args.beam
        Returns:
            nbest_hyps:
        """
        # search params
        beam = 2
        nbest = 3
        maxlen = 4
        self.sos_id = self.eos_id = 1
        # if args.decode_max_len == 0:
        #     maxlen = encoder_outputs.size(0)
        # else:
        #     maxlen = args.decode_max_len

        encoder_outputs = encoder_outputs.unsqueeze(0)

        # prepare sos
        ys = torch.ones(1, 1).fill_(self.sos_id).type_as(encoder_outputs).long()

        # yseq: 1xT
        hyp = {'score': 0.0, 'yseq': ys}
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                ys = hyp['yseq']  # 1 x i

                dec_output = self.forward(ys, h, mask).squeeze(0)
                local_scores = F.log_softmax(dec_output, dim=1)
                # topk scores
                local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)
                for j in range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = torch.ones(1, (1 + ys.size(1))).type_as(encoder_outputs).long()
                    new_hyp['yseq'][:, :ys.size(1)] = hyp['yseq']
                    new_hyp['yseq'][:, ys.size(1)] = int(local_best_ids[0, j])
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(hyps_best_kept,
                                        key=lambda x: x['score'],
                                        reverse=True)[:beam]
                print(hyps_best_kept,'kept')
            # end for hyp in hyps
            hyps = hyps_best_kept

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                for hyp in hyps:
                    hyp['yseq'] = torch.cat([hyp['yseq'],
                                             torch.ones(1, 1).fill_(self.eos_id).type_as(encoder_outputs).long()],
                                            dim=1)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][0, -1] == self.eos_id:
                    ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            hyps = remained_hyps
            if len(hyps) > 0:
                print('remained hypothes: ' + str(len(hyps)))
            else:
                print('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                # print('hypo: ' + ''.join([char_list[int(x)]
                #                           for x in hyp['yseq'][0, 1:]]))
                print('hyp', hyp)
        # end for i in range(maxlen)
        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[
                     :min(len(ended_hyps), nbest)]
        # compitable with LAS implementation
        for hyp in nbest_hyps:
            hyp['yseq'] = hyp['yseq'][0].cpu().numpy().tolist()
        return nbest_hyps


if __name__ == '__main__':
    from mask import *
    inp = torch.ones((3, 7)).long()
    h = torch.rand((1, 8, 10))
    hm = torch.zeros((1, 1, 8))
    dec = Decoder(4, 1, 1, 10, 10, 0)
    out = dec.recognize_beam(h, hm, 0, 0)
    print(out)
