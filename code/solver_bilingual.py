import click
import random as rd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchtext.vocab as vocab
from statistics import mean
from functools import partial
from sklearn.model_selection import train_test_split
from copy import copy
from data_transfer import Task1Dataset, enrich, enrich_source, enrich_target
from analogy_reg import AnalogyRegressionShared, AnalogyRegressionDiff
from cnn_embeddings import CNNEmbedding
from utils import elapsed_timer, collate, reg_loss_fn, decode
from store_embed_reg import generate_embeddings_file
from eval_reg_merged import test_solver_bilingual_
import calendar
import time
import os


@click.command()
@click.option('--source_language', default="test", help='The language to train the model on.', show_default=True)
@click.option('--target_language', default="test", help='The language to train the model on.', show_default=True)
@click.option('--nb_analogies', default=50000,
              help='The maximum number of analogies (before augmentation) we train the model on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.', show_default=True)
@click.option('--epochs', default=20,
              help='The number of epochs we train the model for.', show_default=True)
@click.option('--loss_mode', default=0,
              help='The mode for the loss.', show_default=True)
@click.option('--rd_seed', default=0,
              help='Random seed when loading the data.', show_default=True)
@click.option('--regression_model_shared', default=False,
              help='Regression model used, if True then shared parameters between the two first linear layers.', show_default=True)
def train_solver(source_language, target_language, nb_analogies, epochs, loss_mode, rd_seed, regression_model_shared):
    '''Trains an analogy solving model for a given language.

    Arguments:
    source_language -- The language of the data to use for the embedding model and the first part of the analogies during the training.
    target_language -- The language of the data to use for the second part of the analogies during the training.
    nb_analogies -- The number of analogies to use (before augmentation) for the training.
    epochs -- The number of epochs we train the model for.
    loss_mode -- The mode we use for the loss, see reg_loss_fn function.
    rd_seed -- The seed for the random module.'''

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rd.seed(rd_seed)

    path_source_language = f"models/classification_8x8/classification_balanced_CNN_{source_language}_20e.pth"
    saved_data_embed_source = torch.load(path_source_language)
    source_voc = saved_data_embed_source['word_voc_id']
    target_voc = saved_data_embed_source['word_voc_id']



    ## Train dataset

    if source_language == 'japanese':
        train_dataset = Task1Dataset(source_language=source_language, target_language=target_language, word_voc=saved_data_embed_source['word_voc'], mode="train", word_encoding="char")
        japanese_train_analogies, japanese_test_analogies = train_test_split(train_dataset.analogies, test_size=0.3, random_state = 42)
        train_dataset.analogies = japanese_train_analogies
    else:
        train_dataset = Task1Dataset(source_language=source_language, target_language=target_language, word_voc=saved_data_embed_source['word_voc'], mode="train", word_encoding="char")

    if not len(train_dataset):
        print('Aborted: empty dataset\n')
        return None
    voc = saved_data_embed_source['word_voc_id']
    BOS_ID = len(voc) # (max value + 1) is used for the beginning of sequence value
    EOS_ID = len(voc) + 1 # (max value + 2) is used for the end of sequence value

    # Get subsets

    if len(train_dataset) > nb_analogies:
        train_indices = list(range(len(train_dataset)))
        train_sub_indices = rd.sample(train_indices, nb_analogies)
        train_subset = Subset(train_dataset, train_sub_indices)
    else:
        train_subset = train_dataset



    # Load data
    train_dataloader = DataLoader(train_subset, shuffle = True, collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID))


    # --- Training models ---
    if source_language == 'japanese':
        emb_size = 512
    else:
        emb_size = 64

    if regression_model_shared:
        regression_model = AnalogyRegressionShared(emb_size=16*5) # 16 because 16 filters of each size, 5 because 5 sizes
    else:
        regression_model = AnalogyRegressionDiff(emb_size=16*5)

    embedding_model = CNNEmbedding(emb_size=emb_size, voc_size = len(source_voc) + 2)
    embedding_model.load_state_dict(saved_data_embed_source['state_dict_embeddings'])
    embedding_model.eval()


    # --- Training Loop ---
    embedding_model.to(device)
    regression_model.to(device)

    optimizer = torch.optim.Adam(list(regression_model.parameters()) + list(embedding_model.parameters()), lr=1e-3)
    criterion = nn.MSELoss()

    losses_list = []
    times_list = []

    for epoch in range(epochs):

        if epoch == 0:
            for param in embedding_model.parameters():
                param.requires_grad = False
        elif epoch == epochs//2:
            for param in embedding_model.parameters():
                param.requires_grad = True

        losses = []
        with elapsed_timer() as elapsed:

            for a, b, c, d in train_dataloader:

                optimizer.zero_grad()

                # compute the embeddings
                a = embedding_model(a.to(device))
                b = embedding_model(b.to(device))
                c = embedding_model(c.to(device))
                d = embedding_model(d.to(device))

                # to be able to add other losses, which are tensors, we initialize the loss as a 0 tensor
                loss = torch.tensor(0).to(device).float()

                for a, b, c, d in enrich(a, b, c, d):

                    d_pred = regression_model(a, b, c)

                    loss += reg_loss_fn(a, b, c, d, d_pred, loss_mode)

                loss.backward()
                optimizer.step()

                losses.append(loss.cpu().item())

        losses_list.append(mean(losses))
        times_list.append(elapsed())
        print(f"Epoch: {epoch}, Run time: {times_list[-1]:4.5}s, Loss: {losses_list[-1]}")

    path_models = f"models/bilingual/regression_cnn_{source_language}_to_{target_language}_{epochs}e_{loss_mode}mode_seed{rd_seed}.pth"
    torch.save({"state_dict": regression_model.cpu().state_dict(), 'state_dict_embeddings': embedding_model.cpu().state_dict(), "losses": losses_list, "times": times_list, 'voc': saved_data_embed_source['word_voc'], 'voc_id': saved_data_embed_source['word_voc_id']}, path_models)




    # --- Test models ---

    save_dir = f"regression/bilingual_seed{rd_seed}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    print()

    r = test_solver_bilingual_(language1=source_language,
                                language2=target_language,
                                nb_analogies=nb_analogies,
                                path_models=path_models,
                                folder=save_dir)

    with open(f"regression/bilingual_results.txt", 'a') as f:
        try:
            f.write(f"{rd_seed};{source_language};{target_language};{r['analogies_tested']};{'{:.2f}'.format(r['acc']*100)};{'{:.2f}'.format(r['acc_words']*100)}\n")
        except ValueError:
            f.write(f"{rd_seed};{source_language};{target_language};{r['analogies_tested']};{r['acc']};{r['acc_words']}\n")

if __name__ == '__main__':
    train_solver()
