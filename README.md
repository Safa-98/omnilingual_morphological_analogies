# A Decoder for Morphological Embeddings
## Description
In this project, we aim to build a decoder for morphological embeddings that transforms embedded vectors to words. We will be working on morphological inflections and dervivations like suffixes, prefixes, etc. For the datasets, we will work with the Sigmorphon2016 data file, which contains 10 languages, and the Japanese dataset.

### Setting-Up the Japanese Data
The Japanese data is stored as a Sigmorphon2016 data file `japanese-task1-train` in the code folder, and should be moved to `sigmorphon2016/data`, the Sigmorphon2016 data folder. This can be done by running the following command:

```bash
mv code/japanese-task1-train sigmorphon2016/data/
```


## Files and Folders
- `data.py`: tools to load Sigmorphon2016 datasets, contains the main dataset class `Task1Dataset` and the data augmentation functions `enrich` and `generate_negative`
- `cnn_embeddings.py`: neural network to embed words (the encoder)
- `store_cnn_embeddings.py`: functions to store the embeddings of the train and test set of a given language
- `utils.py`: tools for the different codes


