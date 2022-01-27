#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import click
import torch, torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from li_dataloader_data import MyDataModule
from li_classif import LitEmb


@click.command()
@click.option('--language', default="japanese", help='The language to train the model on.', show_default=True)
@click.option('--max_epochs', default=20,
              help='The maximum number of epochs we train the model for.', show_default=True)
@click.option('--nb_analogies', default=50000,
             help='The maximum number of analogies (before augmentation) we train the model on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.', show_default=True)


def train_classifier(language, nb_analogies, max_epochs):
    '''Trains a decoder based on word embeddings.

    Arguments:
    language -- The language of the data to use for the training.
    nb_analogies -- The (maximal) number of words to use for the training and testing (limited by the size of the dataset).
    epochs -- The number of epochs we train the model for.'''

    num_workers=0
    persistent_workers=False
    mode="train"

    # --- Define models ---
    if language == 'japanese':
        emb_size = 512
    else:
        emb_size = 64

    torch.multiprocessing.set_sharing_strategy('file_system')

     # model
    dm = MyDataModule(language=language, nb_analogies=nb_analogies, mode=mode,num_workers=num_workers,persistent_workers=persistent_workers) #language='arabic', batch_size=1000, max_nb_words=10000, num_workers=4
    dm.prepare_data()
    model= LitEmb(emb_size= emb_size, voc_size = len(dm.voc) + 2)


    
    # train
    trainer = pl.Trainer( max_epochs = max_epochs , log_every_n_steps=10,gpus=1, callbacks=[EarlyStopping(monitor="val_pos_accuracy")])
    trainer.fit(model, dm)
                                                                                     
    with open(f'classif/li_{language}.txt', 'w') as f:
        trainer.test(model)


if __name__ == '__main__':
    train_classifier()

