from config import VOCAB_PATH
from data import LANGUAGES, Task1Dataset, enrich

for lang in LANGUAGES:
    dataset = Task1Dataset(language=lang, mode="train", word_encoding=None)
    voc = dataset.get_vocab()

    if lang != "japanese":
        dataset = Task1Dataset(language=lang, mode="test", word_encoding=None)
        voc.update(dataset.get_vocab())

    voc = list(voc)
    voc.sort()

    with open(VOCAB_PATH.format(language = lang), "w") as f:
        f.write('\n'.join(voc))

    print(f"Wrote {len(voc)} words for the vocabulary of {lang}.")