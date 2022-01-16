import random as rd
from copy import copy
from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, random_split

from analogy_reg import AnalogyRegression
from cnn_embeddings import CNNEmbedding
from config import JAP_SPLIT_RANDOM_STATE, VOCAB_PATH
from data import LANGUAGES, Task1Dataset, enrich
from utils import collate, pad

MODE = 0

class Reg(pl.LightningModule):
    def __init__(self, char_emb_size, id_to_char):
        super().__init__()
        self.save_hyperparameters()
        self.reg = AnalogyRegression(80)
        self.emb = CNNEmbedding(char_emb_size, len(id_to_char) + 2)
        self.id_to_char = id_to_char
        self.char_to_id = {c: i for i, c in enumerate(id_to_char)}

        self.criterion = nn.MSELoss()
        self.cosine_embedding_loss = nn.CosineEmbeddingLoss(margin=0.5)

    def encode_fn(self, w):
        return torch.LongTensor([self.char_to_id.get(c, -1) for c in w])

    def loss_fn(self, a, b, c, d, d_pred):
        if MODE == 0:
            return self.cosine_embedding_loss(
                    torch.cat([d_pred]*4, dim=0),
                    torch.cat([d,a,b,c], dim=0),
                    torch.cat([torch.ones(a.size(0)), -torch.ones(a.size(0) * 3)]).to(a.device))

        elif MODE == 1:
            good = self.criterion(d, d_pred)
            bad = self.criterion(d[torch.randperm(d.size(0))], d_pred)

            return (good + 1) / (bad + 1)

        else:
            return (1 + self.criterion(d_pred, d) * 6) / (1 +
                self.criterion(a,b) +
                self.criterion(a,c) +
                self.criterion(a,d) +
                self.criterion(b,c) +
                self.criterion(b,d) +
                self.criterion(c,d))


    def configure_optimizers(self):
        # @lightning method
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        self.emb_prepared = False

        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a = self.emb(a)
        b = self.emb(b)
        c = self.emb(c)
        d = self.emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)

        for a_, b_, c_, d_ in enrich(a, b, c, d):
            d_pred = self.reg(a_, b_, c_)

            loss += self.loss_fn(a_, b_, c_, d_, d_pred)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a = self.emb(a)
        b = self.emb(b)
        c = self.emb(c)
        d = self.emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)

        for a_, b_, c_, d_ in enrich(a, b, c, d):
            d_pred = self.reg(a_, b_, c_)

            loss += self.loss_fn(a_, b_, c_, d_, d_pred)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # @lightning method
        a,b,c,d = batch

        # compute the embeddings
        a = self.emb(a)
        b = self.emb(b)
        c = self.emb(c)
        d = self.emb(d)

        loss = torch.tensor(0, device=a.device, dtype=float)
        acc_cosine = []
        acc_euclid = []

        for a_, b_, c_, d_ in enrich(a, b, c, d):
            d_pred = self.reg(a_, b_, c_)

            loss += self.loss_fn(a_, b_, c_, d_, d_pred)

            for d__, d_pred_ in zip(d_, d_pred):
                #print(d__.size(), d_pred_.size(), self.closest_cosine(d__), self.closest_cosine(d_pred_), self.closest_euclid(d__), self.closest_cosine(d_pred_))
                acc_cosine.append(self.closest_cosine(d__) == self.closest_cosine(d_pred_))
                acc_euclid.append(self.closest_euclid(d__) == self.closest_cosine(d_pred_))

        acc_cosine = torch.stack(acc_cosine).float().mean()
        acc_euclid = torch.stack(acc_euclid).float().mean()

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_cosine', acc_cosine, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_euclid', acc_euclid, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def prepare_embeddings(self, lang):
        bos_id, eos_id = len(self.char_to_id), len(self.char_to_id) + 1

        # collate function to create tensors in the right format
        def collate_one(w):
            return pad(self.encode_fn(w), bos_id, eos_id, len(w)+2).view(1,-1)
        
        with open(VOCAB_PATH.format(language=lang), 'r') as f:
            voc = [l.strip() for l in f.readlines() if l.strip()]

        with torch.no_grad():
            vectors = [
                self.emb(collate_one(word)).view(-1) for word in voc
            ]
            vectors = torch.stack(vectors, 0)

        self.register_buffer("val_vectors", vectors, persistent=False)
        self.voc = voc
        self.register_buffer("val_vect_norm", vectors.norm(dim=-1), persistent=False)

        #self.val_vectors, self.val_voc = vectors, voc
        #self.val_vect_norm = self.val_vectors.norm(dim=-1)

        return vectors, voc

    def closest_cosine(self, x, return_index=True):
        numerator = (self.val_vectors * x).sum(dim=-1)
        denominator = self.val_vect_norm * x.norm(dim=-1)
        similarities = numerator / denominator
        if return_index:
            return similarities.argmax()
        else:
            return self.val_vectors.itos[similarities.argmax()]
    def closest_euclid(self, x, return_index=True):
        distances = (self.val_vectors - x).norm(dim=-1)
        if return_index:
            return distances.argmin()
        else:
            return self.val_vectors.itos[distances.argmin()]

def prepare_data(language, nb_analogies, nb_analogies_test, batch_size = 32):
    '''Prepare the data for a given language.

    Arguments:
    language -- The language of the data to use for the training.
    nb_analogies -- The number of analogies to use (before augmentation) for the training.
    nb_analogies_test -- The number of analogies to use (before augmentation) for the testing.'''

    ## Train and test dataset
    train_dataset = Task1Dataset(language=language, mode="train", word_encoding="char")

    if language == "japanese":
        japanese_train_analogies, japanese_test_analogies = train_test_split(train_dataset.analogies, test_size=0.3, random_state = JAP_SPLIT_RANDOM_STATE)
        train_dataset.analogies = japanese_train_analogies
        test_dataset = copy(train_dataset)
        test_dataset.analogies = japanese_test_analogies

        train_dataset.analogies = japanese_train_analogies
    else:
        test_dataset = Task1Dataset(language=language, mode="test", word_encoding="char")

    voc_size = len(train_dataset.word_voc)
    BOS_ID = voc_size # (max value + 1) is used for the beginning of sequence value
    EOS_ID = voc_size + 1 # (max value + 2) is used for the end of sequence value

    # Get subsets
    # sample of datasets to work with, if data has more than a certain no, it selects randomly
    if len(train_dataset) > nb_analogies:
        train_indices = list(range(len(train_dataset)))
        train_sub_indices = rd.sample(train_indices, nb_analogies)
        train_subset = Subset(train_dataset, train_sub_indices)
    else:
        train_subset = train_dataset

    
    if len(test_dataset) > nb_analogies_test:
        test_indices = list(range(len(test_dataset)))
        test_sub_indices = rd.sample(test_indices, nb_analogies_test)
        test_subset = Subset(test_dataset, test_sub_indices)
    else:
        test_subset = test_dataset

    # Load data
    train_length = int(len(train_subset)*.9)
    val_length = len(train_subset) - train_length
    train_data, val_data = random_split(train_subset, lengths=(train_length, val_length))
    train_loader = DataLoader(train_data, 
        collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID),
        num_workers=8,
        shuffle=True,
        batch_size=batch_size,
        persistent_workers=True)
    val_loader = DataLoader(val_data, 
        collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID),
        num_workers=8,
        batch_size=batch_size,
        persistent_workers=True)
    test_loader = DataLoader(test_subset, 
        collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID),
        num_workers=8,
        batch_size=batch_size,
        persistent_workers=True)

    return train_loader, val_loader, test_loader, train_dataset.word_voc

def main(args):
    from pytorch_lightning.plugins import DDPSpawnPlugin
    train_loader, val_loader, test_loader, id_to_char = prepare_data(args.language, args.nb_analogies, args.nb_analogies_test)

    # --- Define models ---
    if args.language == 'japanese':
        char_emb_size = 512
    else:
        char_emb_size = 64

    nn = Reg(char_emb_size=char_emb_size, id_to_char=id_to_char)

    # --- Train model ---
    expe_name = f"reg_univ/{args.language}"
    tb_logger = pl.loggers.TensorBoardLogger('logs/', expe_name, version=args.version)
    trainer = pl.Trainer.from_argparse_args(args,
        callbacks=[EarlyStopping(monitor="val_loss")],
        plugins=DDPSpawnPlugin(find_unused_parameters=False))
    trainer.logger = tb_logger
    trainer.fit(nn, train_loader, val_loader)
    #trainer.test(ckpt_path="best", test_dataloaders=test_dataset)

    with torch.no_grad():
        nn.prepare_embeddings(args.language)
        trainer.test(nn, dataloaders=test_loader)

if __name__ == '__main__':
    from argparse import ArgumentParser

    # argument parsing
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--language', '-l', type=str, default="arabic", help='The language to train the model on.', choices=LANGUAGES)
    parser.add_argument('--nb-analogies', '-n', type=int, default=50000, help='The maximum number of analogies (before augmentation) we train the model on.')
    parser.add_argument('--nb-analogies-test', '-nt', type=int, default=50000, help='The maximum number of analogies (before augmentation) we test the model on.')
    parser.add_argument('--version', '-v', type=str, default="debug", help='The experiment version.')
    args = parser.parse_args()

    main(args)

"""--auto_select_gpus"""
