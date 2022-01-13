import click
import random as rd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchtext.vocab as vocab
from statistics import mean
from functools import partial
from sklearn.model_selection import train_test_split
from copy import copy
import data_merged, data_transfer, data
from analogy_reg import AnalogyRegression
from cnn_embeddings import CNNEmbedding
from utils1 import elapsed_timer, collate
import store_embed_reg_merged, store_embed_reg
import multiprocessing
import calendar
import time
import os


def decode(list_ids, voc):
    return ''.join([voc[i.item()] if i.item() in voc.keys() else '#' for i in list_ids[1:-1]])


@click.command()
@click.option('--language1', default='all', help='The language to train the model on.', show_default=True)
@click.option('--language2', default=None, help='The language to train the model on.', show_default=True)
@click.option('--valid_features', default=[], help='The features we keep for the analogies.', show_default=False)
@click.option('--full_dataset', default=False,
              help='The mode for the loss.', show_default=True)
@click.option('--nb_analogies', default=50000,
              help='The maximum number of analogies (before augmentation) we train the model on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.', show_default=True)
@click.option('--path_models', default="",
             help='The path to the saved embedding and regression models.', show_default=True)
@click.option('--folder', default="",
             help='The path to the saved embedding and regression models.', show_default=True)
def test_solver_omni(language1, language2, valid_features, full_dataset, nb_analogies, path_models, folder):
    test_solver_(language1, language2, valid_features, full_dataset, nb_analogies, path_models, folder)

def test_solver_omni_(language1, language2=None, valid_features=[], full_dataset=False, nb_analogies=50000, path_models="", folder=""):
    # --- Test models ---

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rd.seed(0)

    file_name = folder + f"/{language1.capitalize()}{f'to{language2.capitalize()}' if language2 is not None else ''}.txt"
    path_source_language = f"models/omni_classification_cnn/classification_CNN_omni_augmented_20e.pth"
    saved_data_embed_source = torch.load(path_source_language)
    voc = saved_data_embed_source['voc_id']
    BOS_ID = len(voc) # (max value + 1) is used for the beginning of sequence value
    EOS_ID = len(voc) + 1 # (max value + 2) is used for the end of sequence value

    test_dataset = data_merged.Task1Dataset(language1=language1, language2=language2, valid_features=valid_features, word_voc=saved_data_embed_source['voc'], mode="test", word_encoding="char", full_dataset=full_dataset)
    print(f"Loaded dataset: ", len(test_dataset))
    if not len(test_dataset):
        print("aborted: empty dataset")
        return {'acc': '', 'acc_words': '', 'analogies_tested': 0}

    if len(test_dataset) > nb_analogies:
        test_indices = list(range(len(test_dataset)))
        test_sub_indices = rd.sample(test_indices, nb_analogies)
        test_subset = Subset(test_dataset, test_sub_indices)
    else:
        test_subset = test_dataset

    emb_size = 512
    custom_embeddings_file = folder + f"/full{full_dataset}.txt"
    if not os.path.isfile(custom_embeddings_file):
        custom_embeddings_file = store_embed_reg_merged.generate_embeddings_file(path_embed = path_models, storing_path = custom_embeddings_file, emb_size = emb_size, full_dataset=full_dataset)

    custom_embeddings = vocab.Vectors(name = custom_embeddings_file,
                                      cache = folder,
                                      unk_init = torch.Tensor.normal_)

    custom_embeddings.vectors = custom_embeddings.vectors.to(device)

    saved_models = torch.load(path_models)

    test_dataloader = DataLoader(test_subset, shuffle = True, collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID))

    regression_model = AnalogyRegression(emb_size=16*5) # 16 because 16 filters of each size, 5 because 5 sizes
    regression_model.load_state_dict(saved_models['state_dict'])
    regression_model.eval()
    regression_model.to(device)

    embedding_model = CNNEmbedding(emb_size=emb_size, voc_size = len(voc) + 2)
    embedding_model.load_state_dict(saved_models['state_dict_embeddings'])
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
    #accuracy_euclid = []
    accuracy_cosine_word = []
    #accuracy_euclid_word = []

    decode_voc = {v: k for k,v in voc.items()}

    with open(file_name, 'a') as f:

        with elapsed_timer() as elapsed:

            for a, b, c, d in test_dataloader:

                f.write(f"{decode(a.squeeze(0), decode_voc)}:{decode(b.squeeze(0), decode_voc)}::{decode(c.squeeze(0), decode_voc)}:{decode(d.squeeze(0), decode_voc)}\n")
                #print(f"{decode(a.squeeze(0), decode_voc)}:{decode(b.squeeze(0), decode_voc)}::{decode(c.squeeze(0), decode_voc)}:{decode(d.squeeze(0), decode_voc)}\n")

                # compute the embeddings
                a = embedding_model(a.to(device))
                b = embedding_model(b.to(device))
                c = embedding_model(c.to(device))
                d = embedding_model(d.to(device))

                for a, b, c, d_expected in data_merged.enrich_target(a, b, c, d):#enrich

                    #print("Expected:\n", d_expected.squeeze()[:2])

                    d_pred = regression_model(a, b, c)
                    d_closest_cosine = closest_cosine(d_pred)

                    d_expected_closest_cosine = closest_cosine(d_expected)

                    accuracy_cosine_word.append(d_expected_closest_cosine == d_closest_cosine)

                    f.write(f"\tCOS: {d_expected_closest_cosine} ; {d_closest_cosine}\n")

                    accuracy_cosine.append(torch.allclose(d_expected, custom_embeddings.get_vecs_by_tokens(d_closest_cosine).to(device), atol=1e-03))

            results = {'acc': mean(accuracy_cosine), 'acc_words': mean(accuracy_cosine_word), 'analogies_tested': len(test_dataloader)}
            f.write(f'\n\nRESULTS\n\nAccuracy with Cosine similarity: {results["acc"]}')
            f.write(f'Word accuracy with Cosine similarity: {results["acc_words"]}\n')
            print(f'\nRunning time ({language1.capitalize()}{f"to{language2.capitalize()}" if language2 is not None else ""}): {elapsed():4.5}s')
            print(f'Accuracy with Cosine similarity: {results["acc"]}')
            print(f'Word accuracy with Cosine similarity: {results["acc_words"]}\n')

    return results


@click.command()
@click.option('--language1', default='arabic', help='The language to train the model on.', show_default=True)
@click.option('--language2', default=None, help='The language to train the model on.', show_default=True)
@click.option('--valid_features', default=[], help='The features we keep for the analogies.', show_default=False)
@click.option('--nb_analogies', default=50000,
              help='The maximum number of analogies (before augmentation) we train the model on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.', show_default=True)
@click.option('--path_models', default="",
             help='The path to the saved embedding and regression models.', show_default=True)
@click.option('--folder', default="",
             help='The path to the saved embedding and regression models.', show_default=True)
def test_solver_mono(language1, language2, valid_features, nb_analogies, path_models, folder):
    test_solver_(language1, language2, valid_features, nb_analogies, path_models, folder)

def test_solver_mono_(language1, language2=None, valid_features=[], nb_analogies=50000, path_models="", folder=""):
    # --- Test models ---

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rd.seed(0)

    file_name = folder + f"/{language1.capitalize()}{f'to{language2.capitalize()}' if language2 is not None else ''}.txt"
    path_source_language = f"models/omni_classification_cnn/classification_CNN_omni_augmented_20e.pth"
    saved_data_embed_source = torch.load(path_models)
    voc = saved_data_embed_source['voc_id']

    BOS_ID = len(voc) # (max value + 1) is used for the beginning of sequence value
    EOS_ID = len(voc) + 1 # (max value + 2) is used for the end of sequence value

    if language1 == 'japanese':
        test_dataset = data.Task1Dataset(language=language1, valid_features=valid_features, word_voc=saved_data_embed_source['voc'], mode="train", word_encoding="char")
        japanese_train_analogies, japanese_test_analogies = train_test_split(test_dataset.analogies, test_size=0.3, random_state = 42)
        test_dataset.analogies = japanese_test_analogies
    else:
        test_dataset = data.Task1Dataset(language=language1, valid_features=valid_features, word_voc=saved_data_embed_source['voc'], mode="test", word_encoding="char")
    print(f"Loaded dataset: ", len(test_dataset))
    if not len(test_dataset):
        print("aborted: empty dataset")
        return {'acc': '', 'acc_words': '', 'analogies_tested': 0}

    if len(test_dataset) > nb_analogies:
        test_indices = list(range(len(test_dataset)))
        test_sub_indices = rd.sample(test_indices, nb_analogies)
        test_subset = Subset(test_dataset, test_sub_indices)
    else:
        test_subset = test_dataset

    if language1 == 'japanese':
        emb_size = 512
    else:
        emb_size = 64

    # Store the embeddings in a file
    custom_embeddings_file = folder + f"/{language1}.txt"
    if not os.path.isfile(custom_embeddings_file):
        custom_embeddings_file = store_embed_reg.generate_embeddings_file(language = language1, path_embed = path_models, storing_path = custom_embeddings_file)

    custom_embeddings = vocab.Vectors(name = custom_embeddings_file,
                                      cache = folder,
                                      unk_init = torch.Tensor.normal_)

    custom_embeddings.vectors = custom_embeddings.vectors.to(device)

    saved_models = torch.load(path_models)

    test_dataloader = DataLoader(test_subset, shuffle = True, collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID))

    regression_model = AnalogyRegression(emb_size=16*5) # 16 because 16 filters of each size, 5 because 5 sizes
    regression_model.load_state_dict(saved_models['state_dict'])
    regression_model.eval()
    regression_model.to(device)

    embedding_model = CNNEmbedding(emb_size=emb_size, voc_size = len(voc) + 2)
    embedding_model.load_state_dict(saved_models['state_dict_embeddings'])
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
    accuracy_cosine_word = []

    decode_voc = {v: k for k,v in voc.items()}

    with open(file_name, 'a') as f:

        with elapsed_timer() as elapsed:

            for a, b, c, d in test_dataloader:

                f.write(f"{decode(a.squeeze(0), decode_voc)}:{decode(b.squeeze(0), decode_voc)}::{decode(c.squeeze(0), decode_voc)}:{decode(d.squeeze(0), decode_voc)}\n")
                #print(f"{decode(a.squeeze(0), decode_voc)}:{decode(b.squeeze(0), decode_voc)}::{decode(c.squeeze(0), decode_voc)}:{decode(d.squeeze(0), decode_voc)}\n")

                # compute the embeddings
                a = embedding_model(a.to(device))
                b = embedding_model(b.to(device))
                c = embedding_model(c.to(device))
                d = embedding_model(d.to(device))

                for a, b, c, d_expected in data.enrich(a, b, c, d):

                    #print("Expected:\n", d_expected.squeeze()[:2])

                    d_pred = regression_model(a, b, c)
                    d_closest_cosine = closest_cosine(d_pred)

                    d_expected_closest_cosine = closest_cosine(d_expected)

                    accuracy_cosine_word.append(d_expected_closest_cosine == d_closest_cosine)

                    f.write(f"\tCOS: {d_expected_closest_cosine} ; {d_closest_cosine}\n")

                    accuracy_cosine.append(torch.allclose(d_expected, custom_embeddings.get_vecs_by_tokens(d_closest_cosine).to(device), atol=1e-03))

            results = {'acc': mean(accuracy_cosine), 'acc_words': mean(accuracy_cosine_word), 'analogies_tested': len(test_dataloader)}
            f.write(f'\n\nRESULTS\n\nAccuracy with Cosine similarity: {results["acc"]}')
            f.write(f'Word accuracy with Cosine similarity: {results["acc_words"]}\n')
            print(f'\nRunning time ({language1.capitalize()}{f"to{language2.capitalize()}" if language2 is not None else ""}): {elapsed():4.5}s')
            print(f'Accuracy with Cosine similarity: {results["acc"]}')
            print(f'Word accuracy with Cosine similarity: {results["acc_words"]}\n')

    return results


@click.command()
@click.option('--language1', default='finnish', help='The language to train the model on.', show_default=True)
@click.option('--language2', default=None, help='The language to train the model on.', show_default=True)
@click.option('--nb_analogies', default=50000,
              help='The maximum number of analogies (before augmentation) we train the model on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.', show_default=True)
@click.option('--path_models', default="",
             help='The path to the saved embedding and regression models.', show_default=True)
@click.option('--folder', default="",
             help='The path to the saved embedding and regression models.', show_default=True)
def test_solver_bilingual(language1, language2, nb_analogies, path_models, folder):
    test_solver_(language1, language2, nb_analogies, path_models, folder)

def test_solver_bilingual_(language1, language2=None, nb_analogies=50000, path_models="", folder=""):
    # --- Test models ---

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rd.seed(0)

    file_name = folder + f"/{language1.capitalize()}{f'to{language2.capitalize()}' if language2 is not None else ''}.txt"
    saved_data_embed_source = torch.load(path_models)
    voc = saved_data_embed_source['voc_id']
    BOS_ID = len(voc) # (max value + 1) is used for the beginning of sequence value
    EOS_ID = len(voc) + 1 # (max value + 2) is used for the end of sequence value

    test_dataset = data_transfer.Task1Dataset(source_language=language1, target_language=language2, word_voc=saved_data_embed_source['voc'], mode="test", word_encoding="char")
    print(f"Loaded dataset: ", len(test_dataset))
    if not len(test_dataset):
        print("aborted: empty dataset")
        return {'acc': '', 'acc_words': '', 'analogies_tested': 0}

    if len(test_dataset) > nb_analogies:
        test_indices = list(range(len(test_dataset)))
        test_sub_indices = rd.sample(test_indices, nb_analogies)
        test_subset = Subset(test_dataset, test_sub_indices)
    else:
        test_subset = test_dataset

    if language1 == 'japanese':
        emb_size = 512
    else:
        emb_size = 64

    # Store the embeddings in a file
    custom_embeddings_file = folder + f"/{language1}to{language2}.txt"
    if not os.path.isfile(custom_embeddings_file):
        custom_embeddings_file = store_embed_reg_merged.generate_embeddings_file(path_embed = path_models, storing_path = custom_embeddings_file, emb_size = emb_size)

    custom_embeddings = vocab.Vectors(name = custom_embeddings_file,
                                      cache = folder,
                                      unk_init = torch.Tensor.normal_)

    custom_embeddings.vectors = custom_embeddings.vectors.to(device)

    saved_models = torch.load(path_models)

    test_dataloader = DataLoader(test_subset, shuffle = True, collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID))

    regression_model = AnalogyRegression(emb_size=16*5) # 16 because 16 filters of each size, 5 because 5 sizes
    regression_model.load_state_dict(saved_models['state_dict'])
    regression_model.eval()
    regression_model.to(device)

    embedding_model = CNNEmbedding(emb_size=emb_size, voc_size = len(voc) + 2)
    embedding_model.load_state_dict(saved_models['state_dict_embeddings'])
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
    accuracy_cosine_word = []

    decode_voc = {v: k for k,v in voc.items()}

    with open(file_name, 'a') as f:

        with elapsed_timer() as elapsed:

            for a, b, c, d in test_dataloader:

                f.write(f"{decode(a.squeeze(0), decode_voc)}:{decode(b.squeeze(0), decode_voc)}::{decode(c.squeeze(0), decode_voc)}:{decode(d.squeeze(0), decode_voc)}\n")

                # compute the embeddings
                a = embedding_model(a.to(device))
                b = embedding_model(b.to(device))
                c = embedding_model(c.to(device))
                d = embedding_model(d.to(device))

                for a, b, c, d_expected in data_transfer.enrich_target(a, b, c, d):

                    d_pred = regression_model(a, b, c)
                    d_closest_cosine = closest_cosine(d_pred)

                    d_expected_closest_cosine = closest_cosine(d_expected)

                    accuracy_cosine_word.append(d_expected_closest_cosine == d_closest_cosine)

                    f.write(f"\tCOS: {d_expected_closest_cosine} ; {d_closest_cosine}\n")

                    accuracy_cosine.append(torch.allclose(d_expected, custom_embeddings.get_vecs_by_tokens(d_closest_cosine).to(device), atol=1e-03))

            results = {'acc': mean(accuracy_cosine), 'acc_words': mean(accuracy_cosine_word), 'analogies_tested': len(test_dataloader)}
            f.write(f'\n\nRESULTS\n\nAccuracy with Cosine similarity: {results["acc"]}')
            f.write(f'Word accuracy with Cosine similarity: {results["acc_words"]}\n')
            print(f'\nRunning time ({language1.capitalize()}{f"to{language2.capitalize()}" if language2 is not None else ""}): {elapsed():4.5}s')
            print(f'Accuracy with Cosine similarity: {results["acc"]}')
            print(f'Word accuracy with Cosine similarity: {results["acc_words"]}\n')

    return results

if __name__ == '__main__':
    test_solver()
