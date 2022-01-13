import click
import random as rd
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchtext.vocab as vocab
from statistics import mean
from functools import partial
from sklearn.model_selection import train_test_split
from copy import copy
from data_merged import Task1Dataset, enrich, enrich_source, enrich_target
from analogy_reg import AnalogyRegression
from cnn_embeddings import CNNEmbedding
from utils1 import elapsed_timer, collate
from eval_reg_merged import test_solver_omni_
from store_embed_reg_merged import generate_embeddings_file
import multiprocessing
import calendar
import time
import os


criterion = nn.MSELoss()
cosine_embedding_loss = nn.CosineEmbeddingLoss(margin=0.5)
def reg_loss_fn(a, b, c, d, d_pred, mode=0):
        if mode == 0:
            return cosine_embedding_loss(
                    torch.cat([d_pred]*4, dim=0),
                    torch.cat([d,a,b,c], dim=0),
                    torch.cat([torch.ones(a.size(0)), -torch.ones(a.size(0) * 3)]).to(a.device))

        elif mode == 1:
            good = criterion(d, d_pred)
            bad = criterion(d[torch.randperm(d.size(0))], d_pred)

            return (good + 1) / (bad + 1)

        elif mode == 2:
            return criterion(d_pred, d)

        else:
            return (1 + criterion(d_pred, d) * 6) / (1 +
                criterion(a,b) +
                criterion(a,c) +
                criterion(a,d) +
                criterion(b,c) +
                criterion(b,d) +
                criterion(c,d))

def decode(list_ids, voc):
    return ''.join([voc[i.item()] if i.item() in voc.keys() else '#' for i in list_ids[1:-1]])

@click.command()
@click.option('--languages', default='all', help='The language to train the model on.', show_default=True)
@click.option('--nb_analogies', default=50000,
              help='The maximum number of analogies (before augmentation) we train the model on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.', show_default=True)
@click.option('--epochs', default=20,
              help='The number of epochs we train the model for.', show_default=True)
@click.option('--loss_mode', default=0,
              help='The mode for the loss.', show_default=True)
@click.option('--full_dataset', default=False,
              help='The mode for the loss.', show_default=True)
@click.option('--rd_seed', default=0,
              help='Random seed.', show_default=True)
def train_solver(languages, nb_analogies, epochs, loss_mode, full_dataset, rd_seed):
    '''Trains an analogy solving model for a given language.

    Arguments:
    languages -- The language of the data to use for the training.
    nb_analogies -- The number of analogies to use (before augmentation) for the training.
    epochs -- The number of epochs we train the model for.
    loss_mode -- The mode we use for the loss, see reg_loss_fn function.
    full_dataset -- The training dataset we use, if True then we train on both bilingual and monolingual analogies, else on bilingual ones only.
    rd_seed -- The seed for the random module.'''

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rd.seed(rd_seed)

    gpus = torch.cuda.device_count()
    num_workers = multiprocessing.cpu_count() // 2*gpus if gpus != 0 else multiprocessing.cpu_count()
    expe_version = calendar.timegm(time.gmtime())
    file_name = f"regression/omni_transfer_diff{languages}_full{full_dataset}_{expe_version}_seed{rd_seed}.txt"

    path_source_language = f"models/omni_classification_cnn/classification_CNN_omni_augmented_20e.pth"
    saved_data_embed_source = torch.load(path_source_language)
    source_voc = saved_data_embed_source['voc_id']
    target_voc = saved_data_embed_source['voc_id']



    ## Train dataset

    if languages == 'japanese':
        train_dataset = Task1Dataset(language1=languages, word_voc=saved_data_embed_source['voc'], mode="train", word_encoding="char", full_dataset=full_dataset)
        japanese_train_analogies, japanese_test_analogies = train_test_split(train_dataset.analogies, test_size=0.3, random_state = 42)
        train_dataset.analogies = japanese_train_analogies
    else:
        train_dataset = Task1Dataset(language1=languages, word_voc=saved_data_embed_source['voc'], mode="train", word_encoding="char", full_dataset=full_dataset)

    voc = saved_data_embed_source['voc_id']
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
    emb_size = 512

    regression_model = AnalogyRegression(emb_size=16*5) # 16 because 16 filters of each size, 5 because 5 sizes

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

    path_models = f"models/omni_regression_transfer/omni_regression_transfer_diff{languages}_full{full_dataset}_{epochs}e_{loss_mode}mode_{expe_version}_seed{rd_seed}.pth"
    torch.save({"state_dict": regression_model.cpu().state_dict(), 'state_dict_embeddings': embedding_model.cpu().state_dict(), "losses": losses_list, "times": times_list, 'voc': saved_data_embed_source['voc'], 'voc_id': saved_data_embed_source['voc_id']}, path_models)


    # -- Test model

    save_dir = f"regression/omni_transfer_full{full_dataset}_{expe_version}_seed{rd_seed}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    print()

    # test for each of the five languages and for each pair of language
    LANGUAGES = ["finnish", "german", "hungarian", "spanish", "turkish"]
    results = {language: {} for language in LANGUAGES}
    for language in LANGUAGES:
        print(language.capitalize())
        r = test_solver_omni_(language1 = language,
                    language2 = None,
                    full_dataset = True,
                    nb_analogies = nb_analogies,
                    path_models = path_models,
                    folder = save_dir)
        try:
            results[language][language] = ("{:.2f}".format(r['acc']*100), "{:.2f}".format(r['acc_words']*100), r['analogies_tested'])
        except ValueError:
            results[language][language] = (r['acc'], r['acc_words'], r['analogies_tested'])
    for language1 in LANGUAGES:
        for language2 in LANGUAGES:
            if language1 != language2:
                print(f"{language1.capitalize()} to {language2.capitalize()}")
                r = test_solver_omni_(language1 = language1,
                            language2 = language2,
                            full_dataset = False,
                            nb_analogies = nb_analogies,
                            path_models = path_models,
                            folder = save_dir)
                try:
                    results[language1][language2] = ("{:.2f}".format(r['acc']*100), "{:.2f}".format(r['acc_words']*100), r['analogies_tested'])
                except ValueError:
                    results[language1][language2] = (r['acc'], r['acc_words'], r['analogies_tested'])

    df = pd.DataFrame(results)
    df.to_csv(file_name[:-4]+'.csv', sep=';',float_format="%.2f")

if __name__ == '__main__':
    train_solver()
