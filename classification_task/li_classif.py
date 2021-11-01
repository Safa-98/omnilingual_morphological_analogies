#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import enrich, random_sample_negative, Task1Dataset
from utils import elapsed_timer,  get_accuracy_classification, collate, pad
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from statistics import mean
from torch.utils.data import DataLoader, Subset
from li_dataloader_data import MyDataModule
import torch.multiprocessing
import random as rd
from functools import partial
from copy import copy

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

class LitEmb(pl.LightningModule):
    def __init__(self, emb_size, voc_size):
        ''' Character level CNN word embedding.
        
        It produces an output of length 80 by applying filters of different sizes on the input.
        It uses 16 filters for each size from 2 to 6.
        
        Arguments:
        emb_size -- the size of the input vectors
        voc_size -- the maximum number to find in the input vectors
        '''
        pl.LightningModule.__init__(self)

        self.emb_size = emb_size
        self.voc_size = voc_size

        self.encoder = CNNEmbedding(emb_size, voc_size)
        self.clf = AnalogyClassification(80)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer, "monitor": "train_loss"}

    def training_step(self, batch, batch_idx):
        a, b, c, d = batch
        
        #COMPUTE EMBEDDINGS
        emb_a = self.encoder(a)
        emb_b = self.encoder(b)
        emb_c = self.encoder(c)
        emb_d = self.encoder(d)
    
    
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

        self.log('train_clf_loss', clf_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_clf_acc', clf_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return clf_loss
    
    def validation_step(self, batch, batch_idx):
        a, b, c, d = batch

        emb_a = self.encoder(a)
        emb_b = self.encoder(b)
        emb_c = self.encoder(c)
        emb_d = self.encoder(d)

        emb_a = torch.unsqueeze(emb_a, 1)
        emb_b = torch.unsqueeze(emb_b, 1)
        emb_c = torch.unsqueeze(emb_c, 1)
        emb_d = torch.unsqueeze(emb_d, 1)

        clf_loss = torch.tensor(0).to(batch[0].device).float()
        accuracy_true = []
        accuracy_false = []
        for a_,b_,c_,d_ in enrich(emb_a,emb_b,emb_c,emb_d):
            # positive example, target is 1
            is_analogy = self.clf(a_, b_, c_, d_)

            expected = torch.ones(is_analogy.size(), device=is_analogy.device)
            clf_loss += F.binary_cross_entropy(is_analogy, expected)
            accuracy_true.append(get_accuracy_classification(expected, is_analogy))

        for a_,b_,c_,d_ in random_sample_negative(emb_a,emb_b,emb_c,emb_d):
            # negative example, target is 0
            is_analogy = self.clf(a_, b_, c_, d_)
            expected = torch.zeros(is_analogy.size(), device=is_analogy.device)
            clf_loss += F.binary_cross_entropy(is_analogy, expected)
            accuracy_false.append(get_accuracy_classification(expected, is_analogy))
        
        output= mean(accuracy_true)
        self.log("val_pos_accuracy", output)
        return output
        
    def validation_epoch_end(self, output_results):
        print("\n\nValidation step: ", output_results)
    
    def test_step(self, batch, batch_idx):
        a, b, c, d = batch

        emb_a = self.encoder(a)
        emb_b = self.encoder(b)
        emb_c = self.encoder(c)
        emb_d = self.encoder(d)

        emb_a = torch.unsqueeze(emb_a, 1)
        emb_b = torch.unsqueeze(emb_b, 1)
        emb_c = torch.unsqueeze(emb_c, 1)
        emb_d = torch.unsqueeze(emb_d, 1)

        clf_loss = torch.tensor(0).to(batch[0].device).float()
        accuracy_true = []
        accuracy_false = []
        for a_,b_,c_,d_ in enrich(emb_a,emb_b,emb_c,emb_d):
            # positive example, target is 1
            is_analogy = self.clf(a_, b_, c_, d_)

            expected = torch.ones(is_analogy.size(), device=is_analogy.device)
            clf_loss += F.binary_cross_entropy(is_analogy, expected)
            accuracy_true.append(get_accuracy_classification(expected, is_analogy))
        for a_,b_,c_,d_ in random_sample_negative(emb_a,emb_b,emb_c,emb_d):
            # negative example, target is 0
            is_analogy = self.clf(a_, b_, c_, d_)

            expected = torch.zeros(is_analogy.size(), device=is_analogy.device)
            clf_loss += F.binary_cross_entropy(is_analogy, expected)
            accuracy_false.append(get_accuracy_classification(expected, is_analogy))
        
        output= dict({
                "test_pos_accuracy" : mean(accuracy_true),
                "test_neg_accuracy" : mean(accuracy_false),
            })
        self.log("test", output)
        return output
    
    
    def test_epoch_end(self, output_results):
        positive_accuracy = mean([result['test_pos_accuracy'] for result in output_results])
        negative_accuracy = mean([result['test_neg_accuracy'] for result in output_results])
    
        with open(f'classif/li_{self.trainer.datamodule.language}.txt', 'a') as f:
            f.write(f'\n\n\n Valid: {positive_accuracy}\nInvalid: {negative_accuracy}')

    
if __name__ == '__main__':
    # data
    torch.multiprocessing.set_sharing_strategy('file_system')

    language = 'navajo'
    nb_analogies = 10
    num_workers = 8
    persistent_workers = False

    # --- Define models ---
    if language == 'japanese':
        emb_size = 512
    else:
        emb_size = 64

    # model
    dm = MyDataModule(language=language, nb_analogies=nb_analogies, num_workers=num_workers,persistent_workers=persistent_workers) #language='arabic', batch_size=1000, max_nb_words=10000, num_workers=4
    dm.prepare_data()
    model= LitEmb(emb_size= emb_size, voc_size = len(dm.voc) + 2)
    
    #train
    trainer = pl.Trainer(max_epochs=1,log_every_n_steps=10, callbacks=[checkpoint_callback])#num_processes=4,
    trainer.fit(model, dm)
                  
                    
    with open(f'save_words_{language}.txt', 'w') as f:
        trainer.test(model)



