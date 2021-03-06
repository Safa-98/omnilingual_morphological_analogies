# Towards an omnilingual model for solving morphological analogies


## Description
This project was a continuation of last year's work on ``A Neural Approach to Detecting and Solving Morphological Analogies across Languages``. We initially aimed to build a decoder for morphological embeddings that transforms embedded vectors to words in order to apply it on our previous regression model. We worked on morphological inflections and dervivations like suffixes, prefixes, etc. For the datasets, we used the Sigmorphon2016 data file, which contains 10 languages, and the Japanese dataset. Then, we explored the transferability of our previous regression model between languages by building bilingual analogies. This led us to build a single omnilingual model to solve both monolingual and bilingual analogies. We designed a website using this model.

### Setting-Up the Japanese Data
The Japanese data is stored as a Sigmorphon2016 data file `japanese-task1-train` in the code folder, and should be moved to `sigmorphon2016/data`, the Sigmorphon2016 data folder. This can be done by running the following command:

```bash
mv code/japanese-task1-train sigmorphon2016/data/
```

### Setting-Up the Test Data
We created a very small test dataset, which is very quick to load and makes it easier to check that the code works. It is stored in the code folder and should be moved to `sigmorphon2016/data`, the Sigmorphon2016 data folder. This can be done by running the following command:
The Japanese data is stored as a Sigmorphon2016 data file `japanese-task1-train` in the code folder, and should be moved to `sigmorphon2016/data`, the Sigmorphon2016 data folder. This can be done by running the following commands:

```bash
mv code/test-task1-train sigmorphon2016/data/
mv code/test-task1-train sigmorphon2016/data/
mv code/test-task1-train sigmorphon2016/data/
```


## Files and Folders

### `application` folder
- `main.py`: The code to run to launch the application.
- `update_data.py`: The code to run to update the saved embeddings in case the model or the dataset has changed.

#### `modules` subfolder
Contains all the code and models to make the application work.
##### `interface` subfolder
- `interface.py`: Stores the values received from the web page and produces needed outputs;
- `update_data`: contains the code to store all the embeddings of the words in the vocabulary;
- `data`: contains the analogy dataset and the vocabulary (pickle files).
##### `nn_analogy_solver` subfolder
Contains the code, model and saved embeddings used by the application to solve analogies.

#### `static` subfolder
Contains the css, some fonts and icons (https://fontawesome.com) and a html page with additional information about morphology.

#### `templates` subfolder
Contains the main html template of our web application.


### `code` folder
This folder contains the code we use for this project. The neural networks are based on the Pytorch library, we are in the process of moving all these codes to Pytorch Lightning.
- `data.py`: tools to load Sigmorphon2016 datasets, contains the main dataset class `Task1Dataset` and the data augmentation functions `enrich` and `generate_negative`;
- `data_transfer.py`: similar to `data.py` but for the transfer when solving analogies;
- `cnn_embeddings.py`: neural network to embed words (the encoder), based on CNN;
- `store_cnn_embeddings.py`: functions to store the embeddings of the train and test set of a given language;
- `utils.py`: tools for the different codes;
- `solver_transfer.py`: the code to train and evaluate a bilingual analogy solver model, we solve analogies A:B::C:D where A and B are in the source language and C and D in the target language, we use the embedding model from the source language.

#### `decoder` subfolder
This folder contains the code we used when we worked on the decoder.
- `gru_decoder.py`: neural network to decode words, based on a GRU layer;
- `train_gru_decoder.py`: the code to train the decoder, `cnn_embeddings.py` is used as an encoder (the models we used were trained on an analogy classification task, their parameters are frozen during the training of the decoder)
- `word_data.py`: the Dataset class to load the words of the Sigmorphon2016 dataset together with their embeddings and their representation as a list of IDs;

#### `classification_task` subfolder
This folder contains an updated version of some of the code for the classification task. See https://github.com/AmandineDecker/nn-morpho-analogy for more information.

#### `omnilingual_data` subfolder
This folder contains the datasets for the omnilingual analogy solving task and .csv tables containing the list of features per pair of languages.
- `merged-dev`, `merged-test` and `merged-train`: concatenation of the Finnish, German, Hungarian, Spanish and Turkish datasets where we kept only the lines with features shared by at least two languages;
- `fullmerged-dev`, `fullmerged-test` and `fullmerged-train`: concatenation of the Finnish, German, Hungarian, Spanish and Turkish datasets;
- `features_overlap_dev`, `features_overlap_test` and `features_overlap_train`: tables contatining the list of features for each pair of languages and the number of each feature.

#### `regression_task` subfolder
This folder contains an updated version of some of the code for the analogy solving task. See https://github.com/AmandineDecker/nn-morpho-analogy for more information.


### `articles` folder
This folder contains the papers cited in our report and others that we used for our literature survey.

### `presentations` folder
This folder contains all the slides of our intermediate presentations in a PDF format which each detail the progress of our project, the issues encountered and the future work.

### `report` folder
This folder contains the report of our project.

### `results` folder
The sub-folder `first trials` contains the results we obtained after training our decoder on Arabic and Hungarian. They consist of `.txt` files containing 100 (input word ; generated word) pairs and then 3 metrics: the accuracy, the portion of generated words of same length as the input word, and the mean Levenshtein distance on all the evaluated words. These metrics will be improved afterwards as they are not the most meaningful for our task. Indeed the accuracy does not indicate how far our model is to generate the right word while the mean Levenshtein distance does not take the length of the words into account.
The files names are of the form `save_words_{language of the data}_{number of epochs}e_model_{activation function}_{hidden size of the GRU layer}hsize.txt`.


## Installing the Dependencies

Install Anaconda (or miniconda to save storage space).

Then, create a conda environement (for example `omnilingual_morphological_analogies`) and install the dependencies, using the following commands:

```bash
conda create --name omnilingual_morphological_analogies python=3.9
conda activate omnilingual_morphological_analogies
conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -c=conda-forge
conda install -y numpy scipy pandas scikit-learn click flask waitress
```


## Usage of the code
For all the codes, the parameters have a default value, it is thus not necessary to fill them all.

### Decoder
To train and evaluate a decoder for a language, run ``python train_gru_decoder.py --language=<language> --nb_words=<maximal number of words to use> --epochs=<number of epochs> --activation_function=<activation function to use before the word embedding is fed to the GRU layer, to be chosen between ReLu, Sigmoid and Tanh> --batch_size=<the size of the batches> --hidden_size=<the hidden size of the GRU layer>`` (ex: python train_gru_decoder.py --language=german --nb_words=10000 --epochs=20 --activation_function=sigmoid --batch_size=8 --hidden_size=2048).

### Omnilingual and Monolingual analogy solver models
To train and evaluate a monolingual analogy solver for a language, run ``python solver_monolingual.py --source_language=<language> --nb_analogies=<maximal number of analogies to use> --epochs=<number of epochs>`` (ex: ``python solver_monolingual.py --source_language=german --nb_analogies=50000 --epochs=20``).

To train and evaluate a bilingual analogy solver for a language, run ``python solver_bilingual.py --source_language=<source language> --target_language=<target language> --nb_analogies=<maximal number of analogies to use> --epochs=<number of epochs>`` (ex: ``python solver_bilingual.py --source_language=german --target_language=finnish --nb_analogies=50000 --epochs=20``).

To train and evaluate an omnilingual analogy solver for a language, run ``python solver_omnilingual.py --nb_analogies=<maximal number of analogies to use> --epochs=<number of epochs>`` (ex: ``python solver_omnilingual.py --nb_analogies=50000 --epochs=20``).

### Application
Go to the application folder and run the following command in a command prompt: python3 main.py
If the page does not open directly, open the link displayed: http://0.0.0.0:5000

