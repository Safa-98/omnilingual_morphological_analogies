import torch, torch.nn as nn

class AnalogyRegression(nn.Module):
    def __init__(self, emb_size):
        ''' Regression model to solve an analogy.
        It produces a vector of the save size as the input ones.
        Arguments:
        emb_size -- the size of the input vectors
        '''
        super().__init__()
        self.emb_size = emb_size
        self.ab = nn.Linear(2 * self.emb_size, 2 * self.emb_size)
        self.ac = nn.Linear(2 * self.emb_size, 2 * self.emb_size)
        self.d = nn.Linear(4 * self.emb_size, self.emb_size)

    def forward(self, a, b, c):
        '''Produces d such that a:b::c:d holds.
        Arguments:
        a -- embedding of A
        b -- embedding of B
        c -- embedding of C
        '''
        ab = self.ab(torch.cat([a, b], dim = -1))
        ac = self.ac(torch.cat([a, c], dim = -1))
        d = self.d(torch.cat([ab, ac], dim = -1))
        return d

