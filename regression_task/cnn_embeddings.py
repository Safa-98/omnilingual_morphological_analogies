from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import enrich, random_sample_negative
from utils import collate
from torch.utils.data import Subset

import pytorch_lightning as pl

class CNNEmbedding(nn.Module):
    def __init__(self, emb_size, voc_size):
        ''' Character level CNN word embedding.
        
        It produces an output of length 80 by applying filters of different sizes on the input.
        It uses 16 filters for each size from 2 to 6.
        
        Arguments:
        emb_size -- the size of the input vectors
        voc_size -- the maximum number to find in the input vectors
        '''
        super().__init__()

        self.emb_size = emb_size
        self.voc_size = voc_size

        self.embedding = nn.Embedding(voc_size, emb_size)

        self.conv2 = nn.Conv1d(emb_size, 16, 2, padding = 0)
        self.conv3 = nn.Conv1d(emb_size, 16, 3, padding = 0)
        self.conv4 = nn.Conv1d(emb_size, 16, 4, padding = 1)
        self.conv5 = nn.Conv1d(emb_size, 16, 5, padding = 2)
        self.conv6 = nn.Conv1d(emb_size, 16, 6, padding = 3)


    def forward(self, word):
        # Embedds the word and set the right dimension for the tensor
        unk = word<0
        word[unk] = 0
        word = self.embedding(word)
        word[unk] = 0
        word = torch.transpose(word, 1,2)

        # Apply each conv layer -> torch.Size([batch_size, 16, whatever])
        size2 = self.conv2(word)
        size3 = self.conv3(word)
        size4 = self.conv4(word)
        size5 = self.conv5(word)
        size6 = self.conv6(word)

        # Get the max of each channel -> torch.Size([batch_size, 16])
        maxima2 = torch.max(size2, dim = -1)
        maxima3 = torch.max(size3, dim = -1)
        maxima4 = torch.max(size4, dim = -1)
        maxima5 = torch.max(size5, dim = -1)
        maxima6 = torch.max(size6, dim = -1)

        # Concatenate the 5 vectors to get 1 -> torch.Size([batch_size, 80])
        output = torch.cat([maxima2[0], maxima3[0], maxima4[0], maxima5[0], maxima6[0]], dim = -1)

        return output

class EmbDecoder(nn.Module):
    def __init__(self, emb_size, voc_size):
        ''' Character level CNN word embedding.
        
        It produces an output of length 80 by applying filters of different sizes on the input.
        It uses 16 filters for each size from 2 to 6.
        
        Arguments:
        emb_size -- the size of the input vectors
        voc_size -- the maximum number to find in the input vectors
        '''
        super().__init__()

        self.emb_size = emb_size
        self.voc_size = voc_size

        # Decoder RNN
        self.rnn = nn.GRU(80, emb_size, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(emb_size*2, emb_size, batch_first=True, bidirectional=True)
        self.rnn3 = nn.GRU(emb_size*2, emb_size, batch_first=True, bidirectional=True)
        self.rnn4 = nn.GRU(emb_size*2, emb_size, batch_first=True, bidirectional=True)
        self.out_lin = nn.Linear(emb_size*2, voc_size)


    def forward(self, emb, length=10):
        # emb -> torch.Size([batch_size, 80])
        # Generate RNN input -> torch.Size([batch_size, length, 80])
        input = emb.unsqueeze(1).expand(emb.size(0), length, 80)
        
        # Run RNN -> torch.Size([batch_size, length, emb_size])
        out, _ = self.rnn(input)

        # Run CNN -> torch.Size([batch_size, length, emb_size])
        # out = out.transpose(-1,-2)
        # size2 = self.conv2_(out)
        # size4 = self.conv4_(out)
        # size6 = self.conv6_(out)
        # out = torch.cat([size2,size4,size6], dim=1).transpose(-1,-2)
        out, _ = self.rnn2(torch.relu(out))
        out, _ = self.rnn3(torch.relu(out))
        out, _ = self.rnn4(torch.relu(out))

        # Run CNN -> torch.Size([batch_size, length, voc])
        out = self.out_lin(torch.relu(out))

        return out

class AnalogyClassification(nn.Module):
    def __init__(self, emb_size):
        '''CNN based analogy classifier model.

        It generates a value between 0 and 1 (0 for invalid, 1 for valid) based on four input vectors.
        1st layer (convolutional): 128 filters (= kernels) of size h × w = 1 × 2 with strides (1, 2) and relu activation.
        2nd layer (convolutional): 64 filters of size (2, 2) with strides (2, 2) and relu activation.
        3rd layer (dense, equivalent to linear for PyTorch): one output and sigmoid activation.

        Argument:
        emb_size -- the size of the input vectors'''
        super().__init__()
        self.emb_size = emb_size
        self.conv1 = nn.Conv2d(1, 128, (1,2), stride=(1,2))
        self.conv2 = nn.Conv2d(128, 64, (2,2), stride=(2,2))
        self.linear = nn.Linear(64*(emb_size//2), 1)

    def flatten(self, t):
        '''Flattens the input tensor.'''
        t = t.reshape(t.size()[0], -1)
        return t

    def forward(self, a, b, c, d, p=0):
        """
        
        Expected input shape:
        - a, b, c, d: [batch_size, 1, emb_size]
        """
        image = torch.stack([a, b, c, d], dim = 3)

        # apply dropout
        if p>0:
            image=torch.nn.functional.dropout(image, p)

        x = self.conv1(image)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        output = torch.sigmoid(x)
        return output

class Emb(pl.LightningModule):
    def __init__(self, emb_size, voc_size):
        ''' Character level CNN word embedding.
        
        It produces an output of length 80 by applying filters of different sizes on the input.
        It uses 16 filters for each size from 2 to 6.
        
        Arguments:
        emb_size -- the size of the input vectors
        voc_size -- the maximum number to find in the input vectors
        '''
        pl.LightningModule.__init__(self)

        # Meta
        self.emb_size = emb_size
        self.voc_size = voc_size

        self.encoder = CNNEmbedding(emb_size, voc_size)
        self.decoder = EmbDecoder(emb_size, voc_size)
        self.clf = AnalogyClassification(80)

    def configure_optimizers(self):
        # @lightning method
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return {"optimizer": optimizer, "lr_scheduler": {
            #"scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2),
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            "monitor": "train_loss",
        }}

    def training_step(self, batch, batch_idx):
        # @lightning method
        a, b, c, d = batch

        emb_a = self.encoder(a)
        dec_a = self.decoder(emb_a, a.size(-1))
        emb_b = self.encoder(b)
        dec_b = self.decoder(emb_b, b.size(-1))
        emb_c = self.encoder(c)
        dec_c = self.decoder(emb_c, c.size(-1))
        emb_d = self.encoder(d)
        dec_d = self.decoder(emb_d, d.size(-1))
        recon_loss = F.cross_entropy(dec_a.transpose(-1, -2), a)
        recon_loss += F.cross_entropy(dec_b.transpose(-1, -2), b)
        recon_loss += F.cross_entropy(dec_c.transpose(-1, -2), c)
        recon_loss += F.cross_entropy(dec_d.transpose(-1, -2), d)
        acc = torch.stack([(x == pred_x.argmax(dim=-1)).float().mean() for x, pred_x in ((a,dec_a), (b,dec_b), (c,dec_c), (d,dec_d))]).mean()
        emb_a = torch.unsqueeze(emb_a, 1)
        emb_b = torch.unsqueeze(emb_b, 1)
        emb_c = torch.unsqueeze(emb_c, 1)
        emb_d = torch.unsqueeze(emb_d, 1)

        clf_loss = torch.tensor(0).to(batch[0].device).float()
        clf_acc = []
        for a_,b_,c_,d_ in enrich(emb_a,emb_b,emb_c,emb_d):
            # positive example, target is 1
            is_analogy = self.clf(a_, b_, c_, d_)

            expected = torch.ones(is_analogy.size(), device=is_analogy.device)
            clf_loss += F.binary_cross_entropy(is_analogy, expected)
            clf_acc.append((is_analogy > 0.5).float().mean())
        for a_,b_,c_,d_ in random_sample_negative(emb_a,emb_b,emb_c,emb_d):
            # negative example, target is 0
            is_analogy = self.clf(a_, b_, c_, d_)

            expected = torch.zeros(is_analogy.size(), device=is_analogy.device)
            clf_loss += F.binary_cross_entropy(is_analogy, expected)
            clf_acc.append((is_analogy < 0.5).float().mean())

        clf_acc = torch.stack(clf_acc).mean()

        self.log('train_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_clf_loss', clf_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', recon_loss + clf_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_char_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_clf_acc', clf_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return recon_loss + clf_loss
    def validation_step(self, batch, batch_idx):
        # @lightning method
        a, b, c, d = batch

        emb_a = self.encoder(a)
        dec_a = self.decoder(emb_a, a.size(-1))
        emb_b = self.encoder(b)
        dec_b = self.decoder(emb_b, b.size(-1))
        emb_c = self.encoder(c)
        dec_c = self.decoder(emb_c, c.size(-1))
        emb_d = self.encoder(d)
        dec_d = self.decoder(emb_d, d.size(-1))
        recon_loss = F.cross_entropy(dec_a.transpose(-1, -2), a)
        recon_loss += F.cross_entropy(dec_b.transpose(-1, -2), b)
        recon_loss += F.cross_entropy(dec_c.transpose(-1, -2), c)
        recon_loss += F.cross_entropy(dec_d.transpose(-1, -2), d)
        acc = torch.stack([(x == pred_x.argmax(dim=-1)).float().mean() for x, pred_x in ((a,dec_a), (b,dec_b), (c,dec_c), (d,dec_d))]).mean()
        emb_a = torch.unsqueeze(emb_a, 1)
        emb_b = torch.unsqueeze(emb_b, 1)
        emb_c = torch.unsqueeze(emb_c, 1)
        emb_d = torch.unsqueeze(emb_d, 1)

        clf_loss = torch.tensor(0).to(batch[0].device).float()
        clf_acc = []
        for a_,b_,c_,d_ in enrich(emb_a,emb_b,emb_c,emb_d):
            # positive example, target is 1
            is_analogy = self.clf(a_, b_, c_, d_)

            expected = torch.ones(is_analogy.size(), device=is_analogy.device)
            clf_loss += F.binary_cross_entropy(is_analogy, expected)
            clf_acc.append((is_analogy > 0.5).float().mean())
        for a_,b_,c_,d_ in random_sample_negative(emb_a,emb_b,emb_c,emb_d):
            # negative example, target is 0
            is_analogy = self.clf(a_, b_, c_, d_)

            expected = torch.zeros(is_analogy.size(), device=is_analogy.device)
            clf_loss += F.binary_cross_entropy(is_analogy, expected)
            clf_acc.append((is_analogy < 0.5).float().mean())

        clf_acc = torch.stack(clf_acc).mean()

        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_clf_loss', clf_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', recon_loss + clf_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_char_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_clf_acc', clf_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return recon_loss + clf_loss

from data import Task1Dataset
from argparse import ArgumentParser
from pytorch_lightning import loggers as pl_loggers
from functools import partial

def pad(tensor, bos_id, eos_id, target_size=-1):
    '''Adds a padding symbol at the beginning and at the end of a tensor.

    Arguments:
    tensor -- The tensor to pad.
    bos_id -- The value of the padding for the beginning.
    eos_id -- The value of the padding for the end.'''
    tensor = F.pad(input=tensor, pad=(1,0), mode='constant', value=bos_id)
    tensor = F.pad(input=tensor, pad=(0,1), mode='constant', value=eos_id)
    if target_size > 0 and tensor.size(-1) < target_size:
        tensor = F.pad(input=tensor, pad=(0,target_size - tensor.size(-1)), mode='constant', value=-1)

    return tensor

def collate_(batch, bos_id, eos_id, encoder):
    '''Generates padded tensors for the dataloader.

    Arguments:
    batch -- The original data.
    bos_id -- The value of the padding for the beginning.
    eos_id -- The value of the padding for the end.'''

    len_w = max(len(w) for w in batch)

    batch = torch.stack([pad(encoder(w), bos_id, eos_id, len_w+2) for w in batch])

    return batch

from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def main(args):
    language = "navajo"

    # load data
    train_dataset = Task1Dataset(language=language, mode="train", word_encoding="char")

    # Generate special characters and unify the dictionaries of the training and test sets
    voc = train_dataset.word_voc_id
    BOS_ID = len(voc) # (max value + 1) is used for the beginning of sequence value
    EOS_ID = len(voc) + 1 # (max value + 2) is used for the end of sequence value

    vocab = list(train_dataset.get_vocab())
    print(len(vocab))
    train_loader = torch.utils.data.DataLoader(Subset(train_dataset,range(1000)),#vocab[:1024],
        collate_fn=partial(collate, bos_id = BOS_ID, eos_id = EOS_ID),
        num_workers=8,
        shuffle=True,
        batch_size=32)
    val_loader = torch.utils.data.DataLoader(Subset(train_dataset,range(1000,1100)),#vocab[1024:2048],
        collate_fn=partial(collate, bos_id = BOS_ID, eos_id = EOS_ID),
        num_workers=8,
        shuffle=True,
        batch_size=32)

    # create model
    nn = Emb(64, len(vocab) + 2)

    # train model
    expe_name = f"autoenc_clf/{language}"
    tb_logger = pl_loggers.TensorBoardLogger('logs/', expe_name, version=5)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = tb_logger
    trainer.fit(nn, train_loader, val_loader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)