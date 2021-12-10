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
from analogy_reg import AnalogyRegression
from cnn_embeddings import CNNEmbedding
from utils1 import elapsed_timer, collate
from store_embed_reg import generate_embeddings_file
import multiprocessing
import calendar
import time


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
@click.option('--source_language', default="test", help='The language to train the model on.', show_default=True)
@click.option('--target_language', default="test", help='The language to train the model on.', show_default=True)
@click.option('--nb_analogies', default=50000,
              help='The maximum number of analogies (before augmentation) we train the model on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.', show_default=True)
@click.option('--epochs', default=20,
              help='The number of epochs we train the model for.', show_default=True)
@click.option('--loss_mode', default=0,
              help='The mode for the loss.', show_default=True)
def train_solver(source_language, target_language, nb_analogies, epochs, loss_mode):
    '''Trains an analogy solving model for a given language.

    Arguments:
    source_language -- The source language of the data to use (words A and B in A:B::C:D).
    target_language -- The target language of the data to use (words C and D in A:B::C:D).
    nb_analogies -- The number of analogies to use (before augmentation) for the training.
    epochs -- The number of epochs we train the model for.
    loss_mode -- The mode we use for the loss.'''

    device = "cuda" if torch.cuda.is_available() else "cpu"

    gpus = torch.cuda.device_count()
    num_workers = multiprocessing.cpu_count() // 2*gpus if gpus != 0 else multiprocessing.cpu_count()
    expe_version = calendar.timegm(time.gmtime())
    file_name = f"regression/old_{source_language}_to_{target_language}_{expe_version}.txt"

    path_source_language = f"models/classification_8x8/classification_balanced_CNN_{source_language}_20e.pth"
    saved_data_embed_source = torch.load(path_source_language)
    source_voc = saved_data_embed_source['word_voc_id']
    target_voc = saved_data_embed_source['word_voc_id']

    train_dataset = Task1Dataset(source_language=source_language, target_language=target_language, word_voc=saved_data_embed_source['word_voc'], mode="train", word_encoding="char")
    test_dataset = Task1Dataset(source_language=source_language, target_language=target_language, word_voc=saved_data_embed_source['word_voc'], mode="test", word_encoding="char")

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

    if len(test_dataset) > nb_analogies:
        test_indices = list(range(len(test_dataset)))
        test_sub_indices = rd.sample(test_indices, nb_analogies)
        test_subset = Subset(test_dataset, test_sub_indices)
    else:
        test_subset = test_dataset

    # Load data
    train_dataloader = DataLoader(train_subset, shuffle = True, collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID))
    test_dataloader = DataLoader(test_subset, shuffle = True, collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID))


    # --- Training models ---
    if source_language == 'japanese':
        emb_size = 512
    else:
        emb_size = 64

    regression_model = AnalogyRegression(emb_size=16*5) # 16 because 16 filters of each size, 5 because 5 sizes

    #print(len(voc))
    embedding_model = CNNEmbedding(emb_size=emb_size, voc_size = len(source_voc) + 2)
    embedding_model.load_state_dict(saved_data_embed_source['state_dict_embeddings'])
    embedding_model.eval()


    # --- Training Loop ---
    embedding_model.to(device)
    regression_model.to(device)

    optimizer = torch.optim.Adam(regression_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    losses_list = []
    times_list = []

    for epoch in range(epochs):

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

                #data = torch.stack([a, b, c, d], dim = 1)

                for a, b, c, d in enrich(a, b, c, d):

                    d_pred = regression_model(a, b, c)

                    #loss += criterion(d_pred, d)
                    loss += reg_loss_fn(a, b, c, d, d_pred, loss_mode)

                loss.backward()
                optimizer.step()

                losses.append(loss.cpu().item())

        losses_list.append(mean(losses))
        times_list.append(elapsed())
        print(f"Epoch: {epoch}, Run time: {times_list[-1]:4.5}s, Loss: {losses_list[-1]}")

    torch.save({"state_dict": regression_model.cpu().state_dict(), "losses": losses_list, "times": times_list}, f"models/regression_old/regression_cnn_{source_language}_to_{target_language}_{epochs}e_{loss_mode}mode.pth")




    # --- Test models ---

    # Store the embeddings in a file
    path_embed = f"models/classification_8x8/classification_balanced_CNN_{source_language}_20e.pth"
    custom_embeddings_file = generate_embeddings_file(language = target_language, path_embed = path_embed, storing_path = f"regression/saved_embeddings_old/{source_language}_to_{target_language}_{loss_mode}mode.txt")

    custom_embeddings = vocab.Vectors(name = custom_embeddings_file,
                                      cache = 'regression/saved_embeddings_old',
                                      unk_init = torch.Tensor.normal_)

    custom_embeddings.vectors = custom_embeddings.vectors.to(device)

    regression_model.eval()
    regression_model.to(device)
    embedding_model.eval()
    embedding_model.to(device)

    # Cosine distance
    stored_lengths = torch.sqrt((custom_embeddings.vectors ** 2).sum(dim=1))

    def closest_cosine(vec):
        numerator = (custom_embeddings.vectors * vec).sum(dim=1)
        denominator = stored_lengths * torch.sqrt((vec ** 2).sum())
        similarities = numerator / denominator
        return custom_embeddings.itos[similarities.argmax()]

    # Euclidian distance
    def closest_euclid(vec):
        dists = torch.sqrt(((custom_embeddings.vectors - vec) ** 2).sum(dim=1))
        return custom_embeddings.itos[dists.argmin()]

    regression_model.to(device)
    embedding_model.to(device)

    accuracy_cosine = []
    accuracy_euclid = []
    #accuracy_cosine_word = []
    #accuracy_euclid_word = []

    decode_voc = {v: k for k,v in voc.items()}

    with open(file_name, 'w') as f:

        with elapsed_timer() as elapsed:

            for a, b, c, d in test_dataloader:

                f.write(f"{decode(a.squeeze(0), decode_voc)}:{decode(b.squeeze(0), decode_voc)}::{decode(c.squeeze(0), decode_voc)}:{decode(d.squeeze(0), decode_voc)}\n")

                # compute the embeddings

                a = embedding_model(a.to(device))
                b = embedding_model(b.to(device))
                c = embedding_model(c.to(device))
                d = embedding_model(d.to(device))

                #data = torch.stack([a, b, c, d], dim = 1)

                for a, b, c, d_expected in enrich_target(a, b, c, d):

                    d_pred = regression_model(a, b, c)
                    d_closest_cosine = closest_cosine(d_pred)
                    d_closest_euclid = closest_euclid(d_pred)

                    d_expected_closest_cosine = closest_cosine(d_expected)
                    d_expected_closest_euclid = closest_euclid(d_expected)

                    #accuracy_cosine_word.append(d_expected_closest_cosine == d_closest_cosine)
                    #accuracy_euclid_word.append(d_expected_closest_euclid == d_closest_euclid)

                    f.write(f"\tCOS: {d_expected_closest_cosine} ; {d_closest_cosine}\n\tEUC: {d_expected_closest_euclid} ; {d_closest_euclid}\n")


                    accuracy_cosine.append(torch.allclose(d_expected, custom_embeddings.get_vecs_by_tokens(d_closest_cosine).to(device), atol=1e-03))
                    accuracy_euclid.append(torch.allclose(d_expected, custom_embeddings.get_vecs_by_tokens(d_closest_euclid).to(device), atol=1e-03))

        f.write(f'\n\nAccuracy with Cosine similarity: {mean(accuracy_cosine)}\nAccuracy with Euclidean distance: {mean(accuracy_euclid)}\n')
        #f.write(f'Word accuracy with Cosine similarity: {mean(accuracy_cosine_word)}\nWord accuracy with Euclidean distance: {mean(accuracy_euclid_word)}\n\n')
        print(f'\n\nAccuracy with Cosine similarity: {mean(accuracy_cosine)}\nAccuracy with Euclidean distance: {mean(accuracy_euclid)}\n')
        #print(f'Word accuracy with Cosine similarity: {mean(accuracy_cosine_word)}\nWord accuracy with Euclidean distance: {mean(accuracy_euclid_word)}\n\n')

if __name__ == '__main__':
    train_solver()
