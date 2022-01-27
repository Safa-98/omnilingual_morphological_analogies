import torch, torch.nn as nn
from modules.nn_analogy_solver.analogy_reg import AnalogyRegression
import torchtext.vocab as vocab
from modules.nn_analogy_solver.cnn_embeddings import CNNEmbedding

class Solver():
    def __init__(self):
        '''Deals with the neural models to solve analogies.
        '''
        self.model = torch.load('modules/nn_analogy_solver/models/omnilingual_solver.pth')

        self.encode_voc = self.model['voc_id']
        self.decode_voc = {v: k for k,v in self.encode_voc.items()}
        self.voc_chars = self.encode_voc.keys()
        self.BOS_ID = len(self.decode_voc) # (max value + 1) is used for the beginning of sequence value
        self.EOS_ID = self.BOS_ID + 1

        #languages = [Finnish', 'German', 'Hungarian', 'Spanish', 'Turkish']

        # the models
        self.embedding_model = CNNEmbedding(emb_size = 512, voc_size = self.BOS_ID + 2)
        self.embedding_model.load_state_dict(self.model['state_dict_embeddings'])
        self.embedding_model.eval()

        self.solving_model = AnalogyRegression(emb_size=16*5)
        self.solving_model.load_state_dict(self.model['state_dict'])
        self.solving_model.eval()

        # the embeddings
        custom_embeddings_file = 'modules/nn_analogy_solver/embeddings/all_vectors.txt'

        self.custom_embeddings = vocab.Vectors(name = custom_embeddings_file,
                                          cache = 'modules/nn_analogy_solver/embeddings',
                                          unk_init = torch.Tensor.normal_)
        # Cosine distance
        self.stored_lengths = torch.sqrt((self.custom_embeddings.vectors ** 2).sum(dim=1))

    def encode(self, word):
        '''Encodes a word characterwise as a list of IDs.
        Arguments:
        word -- The word to encode.
        '''
        return torch.unsqueeze(torch.LongTensor([self.BOS_ID] + [self.encode_voc[c] if c in self.voc_chars else -1 for c in word] + [self.EOS_ID]), 0)

    def closest_cosine(self, vec):
        '''Returns the word whose embedding is the closest to the input.
        Arguments:
        vec -- The input vector.
        '''
        vec = vec.squeeze(0)
        numerator = (self.custom_embeddings.vectors * vec).sum(dim=1)
        denominator = self.stored_lengths * torch.sqrt((vec ** 2).sum())
        similarities = numerator / denominator
        return self.custom_embeddings.itos[similarities.argmax()]

    def solve(self, a,b,c):
        '''Solve the analogy a:b::c:?.
        Arguments:
        a -- Word a of the analogy;
        b -- Word b of the analogy;
        c -- Word c of the analogy.
        '''
        a = self.embedding_model(self.encode(a))
        b = self.embedding_model(self.encode(b))
        c = self.embedding_model(self.encode(c))

        a = torch.unsqueeze(a, 0)
        b = torch.unsqueeze(b, 0)
        c = torch.unsqueeze(c, 0)

        result = self.solving_model(a, b, c)
        closest_word = self.closest_cosine(result)

        return closest_word

if __name__ == '__main__':
    solver = Solver()
    solver.solve('a', 'b', 'c')

