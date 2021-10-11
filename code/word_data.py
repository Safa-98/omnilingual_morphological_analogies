from data import Task1Dataset
from cnn_embeddings import CNNEmbedding
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class WordsDataset(Dataset):
    """
    Dataset of words and their word embeddings.
    Also stores indexed versions of the words using character mappings.
    """
    def __init__(self, language='arabic', mode='train'):

        PATH = "./sigmorphon2016/data/"

        LANGUAGES = ["arabic", "finnish", "georgian", "german", "hungarian", "japanese", "maltese", "navajo", "russian", "spanish", "turkish"]
        assert language in LANGUAGES, f"Language '{language}' is unkown, allowed languages are {LANGUAGES}"
        self.language = language

        MODES = ["train", "dev", "test"]
        assert mode in MODES, f"Mode '{mode}' is unkown, allowed modes are {MODES}"
        self.mode = mode

        # load the data
        filename = f"{language}-task1-{mode}"
        with open(PATH + filename, "r", encoding="utf-8") as f:
            raw_data = [line.strip().split('\t') for line in f]

        # generate word vocabulary
        all_words = set()
        for word_a, feature_b, word_b in raw_data:
            all_words.add(word_a)
            all_words.add(word_b)
        all_words = list(all_words)
        all_words.sort()
        
        # Embedding model
        path = f"classification_balanced20/classification_balanced_CNN_{language}_20e.pth"
        saved_data_embed = torch.load(path)

        # char to ID mapping
        # we could generate it on the training mode in 'generate word vocabulary' with voc.update
        self.char_to_idx = saved_data_embed['word_voc_id']
        self.char_to_idx['BOS'] = len(self.char_to_idx)
        self.char_to_idx['EOS'] = len(self.char_to_idx)
        self.char_to_idx['UNK'] = -1
        self.idx_to_char = {k: v for v,k in self.char_to_idx.items()}
        
        # split Japanese
        if language == 'japanese':
            japanese_train_words, japanese_test_words = train_test_split(all_words, test_size=0.2, random_state = 42)
            japanese_dev_words, japanese_test_words = train_test_split(japanese_test_words, test_size=0.5, random_state = 42)
            if mode == 'train':
                all_words = japanese_train_words
            elif mode == 'dev':
                all_words = japanese_dev_words
            else:
                all_words = japanese_test_words
        
        # Build id to word mapping
        self.idx_to_word = {k: v for k,v in enumerate(all_words)}
        self.word_to_idx = {k: v for v,k in self.idx_to_word.items()}
        
        self.indexed_words = []
        for word in all_words:
            indexed_word = torch.LongTensor(
                [self.char_to_idx['BOS']]+
                [self.char_to_idx[c] if c in self.char_to_idx.keys() else -1 for c in word]+
                [self.char_to_idx['EOS']]
            )

            indexed_word.requires_grad = False

            self.indexed_words.append(indexed_word)
        
        # Embed words
        if language == 'japanese':
            emb_size = 512
        else:
            emb_size = 64
        
        self.embedding_model = CNNEmbedding(emb_size=emb_size, voc_size = len(self.idx_to_char) - 1)
        self.embedding_model.load_state_dict(saved_data_embed['state_dict_embeddings'])
        #self.embedding_model.eval()
        #self.embedding_model = self.embedding_model.to(device)
        self.embedding_model.requires_grad = False

    def __len__(self):
        return len(self.word_to_idx.keys()) #len(self.all_words)

    def __getitem__(self, idx):
        """
        Each sample is a dict that contains 3 elements:
        word: the word as a string
        indexed_word: torch LongTensor contained indices of each char in the word
        embedding: torch Tensor which is the embedding of the word
        """
        # print(idx)
        idx = int(idx)
        #print(self.embed_word_voc_id)
        BOS_ID = self.char_to_idx['BOS']
        EOS_ID = self.char_to_idx['EOS']
        word_tensor = torch.LongTensor([BOS_ID] + [self.char_to_idx[c] if c in self.char_to_idx.keys() else -1 for c in self.idx_to_word[idx]] + [EOS_ID]).unsqueeze(0)
        sample = {'word':self.idx_to_word[idx],
                  'indexed_word':self.indexed_words[idx],
                  'embedding':self.embedding_model(word_tensor)
                 }
        return sample


def collate_words_samples(samples):
    """
    Collates samples from a WordsDataset into a batch. To be used
    as the collate_fn for a dataloader with this dataset.

    Returns a dict with 4 elements:
    words: a list of string representations of the words
    embeddings: a torch Tensor containing each of the word embeddings,
            in the same order as the words list
    packed_input: packed_sequence which serves as input to a decoder model,
            i.e. the START token is appended to the start of the word.
    packed_output: packed_sequence which serves as target of a decoder model,
            i.e. the END token is appended to the end of the word.
    """

    # raw string representation of the words
    # must be sorted to form a packed sequence
    samples = sorted(samples, key=lambda s: -len(s['word']))
    words = [s['word'] for s in samples]

    # hidden state of shape (1, batch_size, hidden_dim).
    # the 1 corresponds to num_layers*num_directions
    embeddings = torch.stack(
        [s['embedding'] for s in samples]
    ).view(1, len(samples), -1)

    input_words = ([s['indexed_word'][:-1] for s in samples])
    output_words = ([s['indexed_word'][1:] for s in samples])

    packed_input = nn.utils.rnn.pack_sequence(input_words)
    packed_output = nn.utils.rnn.pack_sequence(output_words)

    return {'words':words,
            'embeddings':embeddings,
            'packed_input':packed_input,
            'packed_output':packed_output}

if __name__ == "__main__":
    print("WordDataset")
    print(len(WordsDataset()))
    print(WordsDataset()[0])
