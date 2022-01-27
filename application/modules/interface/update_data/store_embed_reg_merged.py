import torch
from modules.nn_analogy_solver.cnn_embeddings import CNNEmbedding
import torch.nn.functional as F
import pickle

def encode_word(voc, word):
    '''Encode a word characterwise as a list of IDs.
    Arguments:
    voc -- Character to ID mapping;
    word -- The word to encode.
    '''
    return [voc[c] if c in voc.keys() else 0 for c in word]

def pad(tensor, bos_id, eos_id):
    '''Pad a tensor with a constant in the beginning and another in the end.
    Arguments:
    tensor -- The tensor to pad.
    bos_id -- The constant to put in the beginning.
    eos_id -- The constant to put in the end.
    '''
    tensor = F.pad(input=tensor, pad=(1,0), mode='constant', value=bos_id)
    tensor = F.pad(input=tensor, pad=(0,1), mode='constant', value=eos_id)
    return tensor

def generate_embeddings_file(path_embed, storing_path, emb_size = 512, full_dataset = False):
    '''Generate a file containing the words of 'full_voc' and their embeddings.
    Meant to be used by torchtext.vocab.
    '''
    saved_data_embed = torch.load(path_embed)

    with open('modules/interface/data/full_voc', 'rb') as f:
        all_words = pickle.load(f)
    voc = saved_data_embed['voc_id']
    BOS_ID = len(voc) # (max value + 1) is used for the beginning of sequence value
    EOS_ID = len(voc) + 1 # (max value + 2) is used for the end of sequence value

    vocabulary = {word: encode_word(voc, word) for word in all_words}

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
