from contextlib import contextmanager
from timeit import default_timer
import torch
import torch.nn.functional as F

@contextmanager
def elapsed_timer():
    """Context manager to easily time executions.

    Usage:
    >>> with elapsed_timer() as t:
    ...     pass # some instructions
    >>> elapsed_time = t()
    """
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def pad(tensor, bos_id, eos_id, target_size=-1):
    '''Adds a padding symbol at the beginning and at the end of a tensor.

    Arguments:
    tensor -- The tensor to pad.
    bos_id -- The value of the padding for the beginning.
    eos_id -- The value of the padding for the end.'''
    tensor = F.pad(input=tensor, pad=(1,0), mode='constant', value=bos_id)
    #print(type(tensor))
    tensor = F.pad(input=tensor, pad=(0,1), mode='constant', value=eos_id)
    if target_size > 0 and tensor.size(-1) < target_size:
        tensor = F.pad(input=tensor, pad=(0,target_size - tensor.size(-1)), mode='constant', value=-1)

    return tensor

def collate(batch, bos_id, eos_id):
    '''Generates padded tensors for the dataloader.

    Arguments:
    batch -- The original data.
    bos_id -- The value of the padding for the beginning.
    eos_id -- The value of the padding for the end.'''

    a_emb, b_emb, c_emb, d_emb = [], [], [], []

    len_a = max(len(a) for a, b, c, d in batch)
    #print(type(len_a))
    len_b = max(len(b) for a, b, c, d in batch)
    len_c = max(len(c) for a, b, c, d in batch)
    len_d = max(len(d) for a, b, c, d in batch)

    for a, b, c, d in batch:
        a_emb.append(pad(a, bos_id, eos_id, len_a+2))
        #print(type(a_emb))
        b_emb.append(pad(b, bos_id, eos_id, len_b+2))
        c_emb.append(pad(c, bos_id, eos_id, len_c+2))
        d_emb.append(pad(d, bos_id, eos_id, len_d+2))

    # make a tensor of all As, af all Bs, of all Cs and of all Ds
    a_emb = torch.stack(a_emb)
    b_emb = torch.stack(b_emb)
    c_emb = torch.stack(c_emb)
    d_emb = torch.stack(d_emb)

    #print(type(a_emb))
    return a_emb, b_emb, c_emb, d_emb

def reg_loss_fn(a, b, c, d, d_pred, mode=0):
    '''Loss function for the regression models.
    '''
        if mode == 0:
            cosine_embedding_loss = nn.CosineEmbeddingLoss(margin=0.5)
            return cosine_embedding_loss(
                    torch.cat([d_pred]*4, dim=0),
                    torch.cat([d,a,b,c], dim=0),
                    torch.cat([torch.ones(a.size(0)), -torch.ones(a.size(0) * 3)]).to(a.device))

        elif mode == 1:
            criterion = nn.MSELoss()
            good = criterion(d, d_pred)
            bad = criterion(d[torch.randperm(d.size(0))], d_pred)

            return (good + 1) / (bad + 1)

        elif mode == 2:
            criterion = nn.MSELoss()
            return criterion(d_pred, d)

        else:
            criterion = nn.MSELoss()
            return (1 + criterion(d_pred, d) * 6) / (1 +
                criterion(a,b) +
                criterion(a,c) +
                criterion(a,d) +
                criterion(b,c) +
                criterion(b,d) +
                criterion(c,d))

def get_accuracy_classification(y_true, y_pred):
    '''Computes the accuracy for a batch of data of the classification task.

    Arguments:
    y_true -- The tensor of expected values.
    y_pred -- The tensor of predicted values.'''
    assert y_true.size() == y_pred.size()
    y_pred = y_pred > 0.5
    if y_pred.ndim > 1:
        return (y_true == y_pred).sum().item() / y_true.size(0)
    else:
        return (y_true == y_pred).sum().item()

def encode_word(voc, word):
    '''Encodes a word into a list of IDs thanks to a character to integer mapping.

    Arguments:
    voc -- The character to integer mapping.
    word -- The word to encode.'''
    return [voc[c] if c in voc.keys() else 0 for c in word]

def decode(list_ids, voc):
    '''Encodes a list of IDs thanks to a integer to character mapping.

    Arguments:
    list_ids -- The list of integers to decode.
    voc -- The integer to character mapping.'''
    return ''.join([voc[i.item()] if i.item() in voc.keys() else '#' for i in list_ids[1:-1]])

def generate_embeddings_file_(train_dataset, test_dataset, embedding_model, language, out_path=None):
    '''Stores the embeddings of the training and test set of a given language and returns the path to the file.

    Arguments:
    language -- Language of the words to store.
    path_embed -- The path to the embedding model to use.'''
    if out_path is None:
        out_path = f"finetune/{language}-vectors.txt"

    vocabulary = {word: encode_word(train_dataset.word_voc_id, word) for word in train_dataset.all_words}
    vocabulary.update({word: encode_word(test_dataset.word_voc_id, word) for word in test_dataset.all_words})

    with open(out_path, 'w') as f:
        for word, embed in vocabulary.items():
            embedding = torch.unsqueeze(torch.LongTensor(embed), 0)
            embedding = embedding_model(pad(embedding, len(train_dataset.word_voc), len(train_dataset.word_voc)+1))
            embedding = torch.squeeze(embedding)
            embedding = embedding.tolist()
            embedding = [str(i) for i in embedding]
            embedding = ' '.join(embedding)
            f.write(f"{word} {embedding}\n")

    return out_path
