#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torch.utils.data import DataLoader, Subset
from utils import elapsed_timer, collate, pad
import pytorch_lightning as pl
from typing import Optional
import random as rd
from functools import partial
from data import Task1Dataset
from sklearn.model_selection import train_test_split
from copy import copy


class MyDataModule(pl.LightningDataModule):
    def __init__(self, language='arabic', mode="train", batch_size=32, nb_analogies=10000, num_workers=4,persistent_workers=False):
        super().__init__()
        self.PATH = "./sigmorphon2016/data/"

        LANGUAGES = ["arabic", "finnish", "georgian", "german", "hungarian", "japanese", "maltese", "navajo", "russian", "spanish", "turkish"]
        MODES = ["train", "dev", "test"]
        assert language in LANGUAGES, f"Language '{language}' is unkown, allowed languages are {LANGUAGES}"
        self.language = language

        assert mode in MODES, f"Mode '{mode}' is unkown, allowed modes are {MODES}"
        self.mode = mode

        self.nb_analogies = nb_analogies
        self.batch_size= batch_size

        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        if self.language == 'japanese':
            assert mode == 'train', f"Mode '{mode}' is unkown for Japanese, the only allowed mode is 'train'"
        filename = f"{language}-task1-{mode}"
        with open(self.PATH + filename, "r", encoding="utf-8") as f:
            [line.strip().split('\t') for line in f]
            

    def prepare_data(self):

        self.train_dataset = Task1Dataset(language=self.language, mode="train", word_encoding="char")
        if self.language == "japanese":
            japanese_train_analogies, japanese_test_analogies = train_test_split(self.train_dataset.analogies, test_size=0.3, random_state = 42)
            japanese_dev_analogies, japanese_test_analogies = train_test_split(japanese_test_analogies, test_size=0.5, random_state = 42)
            self.train_dataset.analogies = japanese_train_analogies
            self.test_dataset = copy(self.train_dataset)
            self.dev_dataset = copy(self.test_dataset)
            self.test_dataset.analogies = japanese_test_analogies
            self.dev_dataset.analogies =  japanese_test_analogies
        else:
            self.test_dataset = Task1Dataset(language=self.language, mode="test", word_encoding="char")

        self.voc = self.train_dataset.word_voc_id
        self.BOS_ID = len(self.voc) # (max value + 1) is used for the beginning of sequence value
        self.EOS_ID = (len(self.voc)+ 1) # (max value + 2) is used for the end of sequence value
        self.test_dataset.word_voc = self.train_dataset.word_voc
        self.test_dataset.word_voc_id = self.voc

        vocab = list(self.train_dataset.get_vocab())
        #print(len(vocab))
         
        
    def train_dataloader(self):
        if len(self.train_dataset) > self.nb_analogies:
            train_indices = list(range(len(self.train_dataset)))
            train_sub_indices = rd.sample(train_indices, self.nb_analogies)
            train_subset = Subset(self.train_dataset, train_sub_indices)
        else:
            train_subset = self.train_dataset
        return DataLoader(train_subset,
                            shuffle=True,
                            collate_fn=partial(collate, bos_id = self.BOS_ID, eos_id = self.EOS_ID),
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            persistent_workers=self.persistent_workers,
                            pin_memory=True,)
    
    def val_dataloader(self):
        if len(self.dev_dataset) > self.nb_analogies:
            dev_indices = list(range(len(self.dev_dataset)))
            dev_sub_indices = rd.sample(dev_indices, self.nb_analogies)
            dev_subset = Subset(self.dev_dataset, dev_sub_indices)
        else:
            dev_subset = self.dev_dataset
        return DataLoader(dev_subset,
                            shuffle=False,
                            collate_fn=partial(collate, bos_id = self.BOS_ID, eos_id = self.EOS_ID),
                            num_workers=self.num_workers,
                            batch_size= 32,
                            persistent_workers=self.persistent_workers,
                            pin_memory=True,)


    def test_dataloader(self):
        if len(self.test_dataset) > self.nb_analogies:
            test_indices = list(range(len(self.test_dataset)))
            test_sub_indices = rd.sample(test_indices, self.nb_analogies)
            test_subset = Subset(self.test_dataset, test_sub_indices)
        else:
            test_subset = self.test_dataset
        return DataLoader(test_subset,
                            shuffle=True,
                            collate_fn=partial(collate, bos_id = self.BOS_ID, eos_id = self.EOS_ID),
                            num_workers=self.num_workers,
                            batch_size=32,
                            persistent_workers=self.persistent_workers,
                            pin_memory=True,)
    
    

