from data_merged import Task1Dataset
import torch, torch.nn as nn
from cnn_embeddings import CNNEmbedding
import torch.nn.functional as F
import numpy
from sklearn.model_selection import train_test_split

def encode_word(voc, word):
    return [voc[c] if c in voc.keys() else 0 for c in word]

def pad(tensor, bos_id, eos_id):
    tensor = F.pad(input=tensor, pad=(1,0), mode='constant', value=bos_id)
    tensor = F.pad(input=tensor, pad=(0,1), mode='constant', value=eos_id)
    return tensor

#language = "german"
#PATH_EMBED = f"pth/classification_CNN_german_20e2.pth"

def generate_embeddings_file(path_embed, storing_path, emb_size = 512, full_dataset = False):

    saved_data_embed = torch.load(path_embed)

    train_dataset = Task1Dataset(word_voc=saved_data_embed['voc'], mode="train", word_encoding="char", full_dataset=full_dataset)
    test_dataset = Task1Dataset(word_voc=saved_data_embed['voc'], mode="test", word_encoding="char", full_dataset=full_dataset)
    #train_dataset.word_voc = saved_data_embed['voc']#word_
    #train_dataset.word_voc_id = saved_data_embed['voc_id']#word_
    voc = saved_data_embed['voc_id']#word_
    BOS_ID = len(voc) # (max value + 1) is used for the beginning of sequence value
    EOS_ID = len(voc) + 1 # (max value + 2) is used for the end of sequence value
    #test_dataset.word_voc = saved_data_embed['voc']#word_
    #test_dataset.word_voc_id = voc

    train_dic = {word: encode_word(train_dataset.word_voc_id, word) for word in train_dataset.all_words}
    test_dic = {word: encode_word(test_dataset.word_voc_id, word) for word in test_dataset.all_words}
    train_dic.update(test_dic)
    vocabulary = train_dic.copy()

    embedding_model = CNNEmbedding(emb_size=emb_size, voc_size = len(voc) + 2)
    embedding_model.load_state_dict(saved_data_embed['state_dict_embeddings'])
    embedding_model.eval()

    #f"embeddings/char_fine/{language}-vectors.txt", 'w'
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
