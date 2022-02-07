# Towards an omnilingual model for solving morphological analogies
## Description
This project was a continuation of last year's work on ``A Neural Approach to Detecting and Solving Morphological Analogies across Languages''. We initially aimed to build a decoder for morphological embeddings that transforms embedded vectors to words in order to apply it on our previous regression model. We worked on morphological inflections and dervivations like suffixes, prefixes, etc. For the datasets, we used the Sigmorphon2016 data file, which contains 10 languages, and the Japanese dataset. Then, we explored the transferability of our previous regression model between languages by building bilingual analogies. This led us to build a single omnilingual model to solve both monolingual and bilingual analogies. We designed a website using this model.

### Setting-Up the Japanese Data
The Japanese data is stored as a Sigmorphon2016 data file `japanese-task1-train` in the code folder, and should be moved to `sigmorphon2016/data`, the Sigmorphon2016 data folder. This can be done by running the following command:

```bash
mv code/japanese-task1-train sigmorphon2016/data/
```

## Files and Folders

### `application` folder

### `code` folder

#### `decoder` subfolder
This folder contains the code we use for this project. The neural networks are based on the Pytorch library, we are in the process of moving all these codes to Pytorch Lightning.
- `data.py`: tools to load Sigmorphon2016 datasets, contains the main dataset class `Task1Dataset` and the data augmentation functions `enrich` and `generate_negative`;
- `data_transfer.py`: similar to `data.py` but for the transfer when solving analogies;
- `cnn_embeddings.py`: neural network to embed words (the encoder), based on CNN;
- `store_cnn_embeddings.py`: functions to store the embeddings of the train and test set of a given language;
- `utils.py`: tools for the different codes;
- `gru_decoder.py`: neural network to decode words, based on a GRU layer;
- `train_gru_decoder.py`: the code to train the decoder, `cnn_embeddings.py` is used as an encoder (the models we used were trained on an analogy classification task, their parameters are frozen during the training of the decoder)
- `word_data.py`: the Dataset class to load the words of the Sigmorphon2016 dataset together with their embeddings and their representation as a list of IDs;
- `solver_transfer.py`: the code to train and evaluate a bilingual analogy solver model, we solve analogies A:B::C:D where A and B are in the source language and C and D in the target language, we use the embedding model from the source language.

#### `classification_task` subfolder

#### `omnilingual_data` subfolder

#### `regression_task` subfolder


### `articles` folder
This folder contains the papers cited in our report and others that we used for our literature survey.

### `presentations` folder
This folder contains all the slides of our intermediate presentations in a PDF format which each detail the progress of our project, the issues encountered and the future work.

### `report` folder
This folder contains the report of our project.

### `results` folder
The sub-folder `first trials` contains the results we obtained after training our decoder on Arabic and Hungarian. They consist of `.txt` files containing 100 (input word ; generated word) pairs and then 3 metrics: the accuracy, the portion of generated words of same length as the input word, and the mean Levenshtein distance on all the evaluated words. These metrics will be improved afterwards as they are not the most meaningful for our task. Indeed the accuracy does not indicate how far our model is to generate the right word while the mean Levenshtein distance does not take the length of the words into account.
The files names are of the form `save_words_{language of the data}_{number of epochs}e_model_{activation function}_{hidden size of the GRU layer}hsize.txt`.

## Usage of the code

### Decoder
To train and evaluate a decoder for a language, run python train_gru_decoder.py --language=<language> --nb_words=<maximal number of words to use> --epochs=<number of epochs> --activation_function=<activation function to use before the word embedding is fed to the GRU layer, to be chosen between ReLu, Sigmoid and Tanh> --batch_size=<the size of the batches> --hidden_size=<the hidden size of the GRU layer> (ex: python train_gru_decoder.py --language=german --nb_words=10000 --epochs=20 --activation_function=relu --batch_size=10 --hidden_size=128).
All of these parameters have a default value, it is thus not necessary to fill them all.

### Omnilingual model

### Application
Go to the application folder and run the following command in a command prompt: python3 main.py
If the page doesn't open directly, open the link displayed: http://0.0.0.0:5000

