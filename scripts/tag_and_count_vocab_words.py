"""
make csv file (e.g. vocab_words/a-a-w-w-w.csv) that contains words in a tokenizer configuration file,
alongside their frequency in corpora of interest (e.g. childes, newsela, wikipedia).

A huggingface tokenizers v0.10 configuration file is expected.
"""
import spacy
import json
from pathlib import Path
import pandas as pd
from zorro import configs

PATH_TOKENIZER = '/home/ph/BabyBERTa/data/tokenizers/a-a-w-w-w-8192.json'
PATH_CORPORA = '/home/ph/BabyBERTa/data/corpora'
DRY_RUN = False

# get vocab from tokenizer, without space symbol
with open(PATH_TOKENIZER) as f:
    tokenizer_data = json.load(f)
vocab = {w.strip(configs.Data.space_symbol) for w in tokenizer_data['model']['vocab'].keys()}

# keep track of which words are excluded - not a candidate for being inserted into test sentences
nds = (configs.Dirs.external_words / "non-dictionary.txt").open().read().split()
nws = (configs.Dirs.external_words / "numbers.txt").open().read().split()
sws = (configs.Dirs.external_words / "stopwords.txt").open().read().split()
excluded_words = set(nds + nws + sws)


def is_excluded(w: str):
    """excluded from being considered as candidate for insertion into test sentences"""
    if w in excluded_words:
        return True
    if w.isdigit():
        return True
    if len(w) == 1:
        return True
    if w.startswith(configs.Data.space_symbol):
        return True
    return False


def init_row(word: str,
             tag: str,
             corpus_initial_: str,
             is_excluded_: bool = False,
             ):
    res = {
        'NN': 1 if tag == 'NN' else 0,
        'NNS': 1 if tag == 'NNS' else 0,
        'JJ': 1 if tag == 'JJ' else 0,
        'total-frequency': 1 if not is_excluded_ else 0,
        'is_excluded': is_excluded_ or is_excluded(word),
    }
    for ci in corpus_names:
        res[f'{ci}-frequency'] = 1 if ci == corpus_initial_ else 0

    return res


nlp = spacy.load('en_core_web_sm')

print(f'Will count vocab words in the following corpora:')
corpus_paths = [p for p in Path(PATH_CORPORA).glob('*.txt')]
for c in corpus_paths:
    print(c)

# get information about all words in corpora
w2row = {}
corpus_names = [c.stem for c in corpus_paths]
for corpus_path, corpus_name in zip(corpus_paths, corpus_names):

    with open(corpus_path) as f:
        sentences = [s for s in f.readlines()]

    for n, sd in enumerate(nlp.pipe(sentences, disable=["parser", "ner"])):
        for sw in sd:
            try:
                w2row[sw.text]['NN'] += 1 if sw.tag_ == 'NN' else 0
                w2row[sw.text]['NNS'] += 1 if sw.tag_ == 'NNS' else 0
                w2row[sw.text]['JJ'] += 1 if sw.tag_ == 'JJ' else 0
                w2row[sw.text]['total-frequency'] += 1
                w2row[sw.text][f'{corpus_name}-frequency'] += 1
            except KeyError:
                w2row[sw.text] = init_row(sw.text, sw.tag_, corpus_name)

        if DRY_RUN and n == 100:
            break

        if n % 1000 == 0:
            print(f'{corpus_path.stem:<24} {n:>12,}/{len(sentences):>12,}')


# ensure each word in df is a word in vocab (including sub-words)
rows = []
words = []
num_words_not_in_corpus = 0
for w in vocab:
    if w not in w2row:
        print(f'Did not find "{w:<20}" in corpus. It may be a sub-word')
        w2row[w] = init_row(w, 'n/a', 'n/a', is_excluded_=True)
        num_words_not_in_corpus += 1
    rows.append(w2row[w])
    words.append(w)
print(f'Did not find {num_words_not_in_corpus} tokens from vocab in corpora')

# make data frame that holds info about each word in vocabZ
df = pd.DataFrame(data=rows, index=words)
df.sort_values(by='total-frequency', inplace=True, ascending=False)

if DRY_RUN:
    print(df)
    exit('Dy run completed after 100 sentences')

# save to csv
num_excluded = 0
out_path = configs.Dirs.data / 'vocab_words' / f'{"-".join(corpus_names)}.csv'
df.to_csv(out_path, index=True)
