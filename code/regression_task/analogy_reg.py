import torch
import torch.nn as nn


class AnalogyRegression(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.ab = nn.Linear(2 * self.emb_size, 2 * self.emb_size)
        self.ac = nn.Linear(2 * self.emb_size, 2 * self.emb_size)
        self.d = nn.Linear(4 * self.emb_size, self.emb_size)

    def forward(self, a, b, c, p=0):

        if p>0:
            a=torch.nn.functional.dropout(a, p)
            b=torch.nn.functional.dropout(b, p)
            c=torch.nn.functional.dropout(c, p)

        ab = self.ab(torch.cat([a, b], dim = -1))
        ac = self.ab(torch.cat([a, c], dim = -1))

        d = self.d(torch.cat([ab, ac], dim = -1))
        return d
