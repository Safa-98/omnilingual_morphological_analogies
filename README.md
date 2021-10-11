# A Decoder for Morphological Embeddings
## Description
In this project, we aim to build a decoder for morphological embeddings that transforms embedded vectors to words. We will be working on morphological inflections and dervivations like suffixes, prefixes, etc. For the datasets, we will work with the Sigmorphon2016 data file, which contains 10 languages, and the Japanese dataset.

### Setting-Up the Japanese Data
The Japanese data is stored as a Sigmorphon2016 data file `japanese-task1-train` in the code folder, and should be moved to `sigmorphon2016/data`, the Sigmorphon2016 data folder. This can be done by running the following command:

```bash
mv code/japanese-task1-train sigmorphon2016/data/
```


## Files and Folders
### `code`folder
This folder contains the code we use for this project. The neural networks are based on the Pytorch library, we are in the process of moving all these codes to Pytorch Lightning.
- `data.py`: tools to load Sigmorphon2016 datasets, contains the main dataset class `Task1Dataset` and the data augmentation functions `enrich` and `generate_negative`;
- `cnn_embeddings.py`: neural network to embed words (the encoder), based on CNN;
- `store_cnn_embeddings.py`: functions to store the embeddings of the train and test set of a given language;
- `utils.py`: tools for the different codes;
- `gru_decoder.py`: neural network to decode words, based on a GRU layer;
- `train_gru_decoder.py`: the code to train the decoder, `cnn_embeddings.py` is used as an encoder (the models we used were trained on an analogy classification task, their parameters are frozen during the training of the decoder)
- `word_data.py`: the Dataset class to load the words of the Sigmorphon2016 dataset together with their embeddings and their representation as a list of IDs;

### `results` folder
The sub-folder `first trials` contain the results we obtained after training our decoder or Arabic and Hungarian. They consist in `.txt` files containing 100 (input word ; generated word) pairs and then 3 metrics: the accuracy, the portion of generated words of same length as the input word, and the mean Levenshtein distance on all the evaluated words. These metrics will be improved afterwards as they are not the most meaningful for our task. Indeed the accuracy does not indicate how far our model is to generate the right word while the mean Levenshtein distance does not take the length of the words into account.
The files names are of the form `save_words_{language of the data}_{number of epochs}e_model_{activation function}_{hidden size of the GRU layer}hsize.txt


## Usage of the code
To train and evaluate a decoder for a language, run python train_gru_decoder.py --language=<language> --nb_words=<maximal number of words to use> --epochs=<number of epochs> --activation_function=<activation function to use before the word embedding is fed to the GRU layer, to be chosen between ReLu, Sigmoid and Tanh> --batch_size=<the size of the batches> --hidden_size=<the hidden size of the GRU layer> (ex: python train_gru_decoder.py --language=german --nb_words=10000 --epochs=20 --activation_function=relu --batch_size=10 --hidden_size=128).
All of these parameters have a default value, it is thus not necessary to fill them all.


