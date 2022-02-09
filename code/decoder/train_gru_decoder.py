import click
import random as rd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchtext.vocab as vocab
from statistics import mean
from functools import partial
from sklearn.model_selection import train_test_split
from copy import copy
from word_data import WordsDataset, collate_words_samples
#from cnn_embeddings import CNNEmbedding
from gru_decoder import CharDecoderRNN
from utils import elapsed_timer, collate
import numpy as np
import Levenshtein as lev


@click.command()
@click.option('--language', default="arabic", help='The language to train the model on.', show_default=True)
@click.option('--nb_words', default=50000,
              help='The maximum number of analogies (before augmentation) we train the model on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.', show_default=True)
@click.option('--epochs', default=20,
              help='The number of epochs we train the model for.', show_default=True)
@click.option('--activation_function', default='relu',
              help='The activation function of the RNN.', show_default=True)
@click.option('--batch_size', default=10,
              help='The number of elements per batch.', show_default=True)
@click.option('--hidden_size', default=128,
              help='The size of the hidden layer.', show_default=True)
def train_decoder(language, nb_words, epochs, activation_function, batch_size, hidden_size):
    '''Trains a decoder based on word embeddings.

    Arguments:
    language -- The language of the data to use for the training.
    nb_words -- The (maximal) number of words to use for the training and testing (limited by the size of the dataset).
    epochs -- The number of epochs we train the model for.
    activation_function -- The activation function applied when the embedding is transformed into a hidden state.
    batch_size -- The number of elements per batch.
    hidden_size -- The size of the hidden layer of the GRU model.'''

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"

    path = f"classification_balanced20/classification_balanced_CNN_{language}_20e.pth"
    saved_data_embed = torch.load(path)

    ## Train dataset

    train_dataset = WordsDataset(language=language, mode='train')
    test_dataset = WordsDataset(language=language, mode='test')

    # Get subsets

    if len(train_dataset) > nb_words:
        train_indices = list(range(len(train_dataset)))
        train_sub_indices = rd.sample(train_indices, nb_words)
        train_subset = Subset(train_dataset, train_sub_indices)
    else:
        train_subset = train_dataset

    if len(test_dataset) > nb_words:
        test_indices = list(range(len(test_dataset)))
        test_sub_indices = rd.sample(test_indices, nb_words)
        test_subset = Subset(test_dataset, test_sub_indices)
    else:
        test_subset = test_dataset

    # Load data
    train_dataloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        # pin_memory=pin,
        collate_fn=collate_words_samples,)
    test_dataloader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=1,
        shuffle=True,
        # pin_memory=pin,
        collate_fn=collate_words_samples,)

    # --- Training models ---
    decoder_model = CharDecoderRNN(hidden_size=hidden_size, char_count=len(train_dataset.idx_to_char.keys()), char_embedding_size = 64, input_embedding_size=80, embedding_to_hidden_activation=activation_function) #'sigmoid', 'tanh'

    # --- Training Loop ---
    #embedding_model.to(device)
    decoder_model.to(device)

    optimizer = torch.optim.Adam(decoder_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses_list = []
    times_list = []

    with open(f"decoder/save_lists_{language}_{epochs}e_{activation_function}_{hidden_size}hsize.txt", 'w') as f_lists:

        for epoch in range(epochs):

            count_saved = 0

            losses = []
            with elapsed_timer() as elapsed:

                for word_batch in train_dataloader:

                    optimizer.zero_grad()
                    loss = torch.tensor(0).to(device).float()

                    # hidden state initialized as embedding
                    hidden = word_batch['embeddings'].to(device)

                    # input and target batches of sequences
                    packed_input = word_batch['packed_input']
                    packed_target = word_batch['packed_output']

                    # forward pass
                    packed_output, hidden = decoder_model(hidden, packed_input)
                    loss += criterion(packed_output.data.to(device), packed_target.data.to(device))

                    if count_saved < 100:
                        produced_ids = [character.argmax().item() for character in packed_output.data]
                        f_lists.write(f"{str(packed_target.data.tolist())} ; {str(produced_ids)}\n")
                        count_saved += 1

                    loss.backward()
                    optimizer.step()

                    losses.append(loss.cpu().item())

            f_lists.write('\n\n')

            losses_list.append(mean(losses))
            times_list.append(elapsed())
            print(f"Epoch: {epoch}, Run time: {times_list[-1]:4.5}s, Loss: {losses_list[-1]}")

    torch.save({"state_dict": decoder_model.cpu().state_dict(), "losses": losses_list, "times": times_list}, f"decoder/decoder_{language}_{epochs}e_{activation_function}_{hidden_size}hsize.pth")

    # --- Evaluation ---

    results_list = []
    equal_length = []
    levenshtein = []

    decoder_model.eval()
    decoder_model.to(device)

    BOS_ID = test_dataset.char_to_idx['BOS'] # BOS from the embedding dictionary
    EOS_ID = test_dataset.char_to_idx['EOS'] # EOS from the decoder dictionary

    with open(f"decoder/save_words_{language}_{epochs}e_model_{activation_function}_{hidden_size}hsize.txt", 'w') as f:

        count_saved = 0

        for word_batch in test_dataloader:

            for idx, word in enumerate(word_batch['words']):

                output_word = []
                output_tensor = []

                word_embedding = word_batch['embeddings'][idx]

                input_tensor = torch.LongTensor([BOS_ID]).to(device) # BOS_ID

                input_tensor = nn.utils.rnn.pack_sequence([input_tensor])

                c, hidden = decoder_model(word_embedding.unsqueeze(0).to(device), input_tensor) # hidden or embedding ?

                last_id = c.data.argmax().item()

                counter = 0
                while last_id != EOS_ID and counter < 30: # EOS_ID

                    output_word.append(test_dataset.idx_to_char[last_id])
                    output_tensor.append(last_id)

                    input_tensor = torch.LongTensor([last_id])
                    input_tensor = nn.utils.rnn.pack_sequence(input_tensor.unsqueeze(0))

                    c, hidden = decoder_model(hidden.squeeze(0).to(device), input_tensor, use_head = False) # hidden or embedding ?

                    last_id = c.data.argmax().item()

                    counter += 1

                if count_saved < 100:
                    f.write(f"{word} ; {''.join(output_word)}\n")
                    count_saved += 1
                results_list.append(word == ''.join(output_word))

                expected_word_idx = test_dataset.word_to_idx[word]
                expected_tensor = test_dataset[expected_word_idx]['indexed_word']
                equal_length.append(len(expected_tensor) == len(output_tensor)+2)
                levenshtein.append(lev.distance(word, ''.join(output_word)))

        f.write('\n\n\n')
        f.write(f"Accuracy: {mean(results_list)}\n")
        f.write(f"Same length:  {sum(equal_length)} / {len(test_dataset)}\n")
        f.write(f"Levenshtein:  {mean(levenshtein)}")

    print(f"Accuracy: {mean(results_list)}")
    print(f"Same length:  {sum(equal_length)} / {len(test_dataset)}")
    print(f"Levenshtein: {mean(results_list)}")

if __name__ == '__main__':
    train_decoder()
