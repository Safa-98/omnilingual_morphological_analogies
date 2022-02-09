from data_merged import Task1Dataset
import torch, torch.nn as nn
from cnn_embeddings import CNNEmbedding
import torch.nn.functional as F
import numpy
from sklearn.model_selection import train_test_split

def encode_word(voc, word):
    '''Encodes a word into a list of IDs thanks to a character to integer mapping.

    Arguments:
    voc -- The character to integer dictionary.
    word -- The word to encode.'''
    return [voc[c] if c in voc.keys() else 0 for c in word]

def pad(tensor, bos_id, eos_id):
    tensor = F.pad(input=tensor, pad=(1,0), mode='constant', value=bos_id)
    tensor = F.pad(input=tensor, pad=(0,1), mode='constant', value=eos_id)
    return tensor

def generate_embeddings_file(path_embed, storing_path, emb_size = 512, full_dataset = False):
    '''Stores the embeddings of the training and test set of a given language and returns the path to the file.

    Arguments:
    path_embed -- The path to the embedding model to use.
    storing_path -- The path where the embeddings will be saved.
    emb_size -- The size of the embedding layer of the embedding model.
    full_dataset -- If True, loads the entire dataset, otherwise loads only the analogies based on features shared by at least two languages.'''

    saved_data_embed = torch.load(path_embed)

    train_dataset = Task1Dataset(word_voc=saved_data_embed['voc'], mode="train", word_encoding="char", full_dataset=full_dataset)
    test_dataset = Task1Dataset(word_voc=saved_data_embed['voc'], mode="test", word_encoding="char", full_dataset=full_dataset)
    voc = saved_data_embed['voc_id']#word_
    BOS_ID = len(voc) # (max value + 1) is used for the beginning of sequence value
    EOS_ID = len(voc) + 1 # (max value + 2) is used for the end of sequence value

    train_dic = {word: encode_word(train_dataset.word_voc_id, word) for word in train_dataset.all_words}
    test_dic = {word: encode_word(test_dataset.word_voc_id, word) for word in test_dataset.all_words}
    train_dic.update(test_dic)
    vocabulary = train_dic.copy()

    embedding_model = CNNEmbedding(emb_size=emb_size, voc_size = len(voc) + 2)
    embedding_model.load_state_dict(saved_data_embed['state_dict_embeddings'])
    embedding_model.eval()

    with open(storing_path, 'w') as f:
        for word, embed in vocabulary.items():
            embedding = torch.unsqueeze(torch.LongTensor(embed), 0)
            embedding = embedding_model(pad(embedding, BOS_ID, EOS_ID))
            embedding = torch.squeeze(embedding)
            embedding = embedding.tolist()
            embedding = [str(i) for i in embedding]
            embedding = ' '.join(embedding)
            f.write(f"{word} {embedding}\n")

    return storing_path
