import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, idim, ffdim, dropout_rate):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(nn.Linear(idim, ffdim),
                                nn.Dropout(dropout_rate),
                                nn.ReLU(),
                                nn.Linear(ffdim, idim))

    def forward(self, x):
        return self.ff(x)
