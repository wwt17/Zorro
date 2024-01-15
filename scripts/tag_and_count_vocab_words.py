"""
make csv file (e.g. vocab_words/babyberta.csv) that contains tokens in a tokenizer configuration file,
alongside their frequency in corpora of interest (e.g. childes, newsela, wikipedia).

A huggingface tokenizers v0.10 configuration file is expected.
"""
import argparse
import spacy
from spacy.tokens import Doc
import json
from pathlib import Path
import pandas as pd
from zorro import configs


POS_TAGS = [
    'NNP',  # proper noun
    'NN' ,
    'NNS',
    'JJ' ,
    'VB' ,  # base form of verb
    'VBD',  # verb past tense
    'VBG',  # verb gerund or present participle
    'VBN',  # verb past participle
    'VBP',  # verb non-3rd person singular present
    'VBZ',  # verb 3rd person singular present
]


def is_excluded(w: str):
    """excluded from being considered as candidate for insertion into test sentences"""
    if w in excluded_words:
        return True
    if w.isdigit():
        return True
    if len(w) == 1:
        return True
    # word must be whole-word in vocab (must have space_symbol).
    # e.g. "phones" may not be in vocab, while its singular form is
    if args.whole_word and f'{configs.Data.space_symbol}{w}' not in vocab:
        return True
    return False


def update_row(
        df: pd.DataFrame,
        idx: int,
        tag: str,
        corpus_name: str,
):
    try:
        df.loc[idx, tag] += 1 # type: ignore
    except KeyError:
        pass
    df.loc[idx, 'total-frequency'] += 1 # type: ignore
    try:
        df.loc[idx, f'{corpus_name}-frequency'] += 1 # type: ignore
    except KeyError:
        pass


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--tokenizer_file", type=Path,
        help="Path to the HuggingFace tokenizer configuration file."
    )
    argparser.add_argument(
        "--vocab_file", type=Path,
        help="Path to the vocab file."
    )
    argparser.add_argument(
        "--corpora", type=Path, nargs="*", required=True,
        help="Paths to the corpora."
    )
    argparser.add_argument(
        "--whole_word", action="store_true",
        help="Ensure remaining words to be whole words."
    )
    argparser.add_argument(
        "--dry_run", action="store_true",
    )
    argparser.add_argument(
        "--dry_run_n_sentences", type=int, default=100
    )
    args = argparser.parse_args()

    if args.vocab_file is not None:
        vocab_name = args.vocab_file.stem
        with open(args.vocab_file) as f:
            vocab = json.load(f)
    elif args.tokenizer_file is not None:
        vocab_name = args.tokenizer_file.stem
        # get vocab from tokenizer, without space symbol
        with open(args.tokenizer_file) as f:
            tokenizer_data = json.load(f)
        vocab = tokenizer_data['model']['vocab']
    else:
        raise Exception("Must provide either tokenizer_file or vocab_file.")
    assert isinstance(vocab, dict)
    assert max(vocab.values()) + 1 == len(vocab)
    vocab_no_space_symbol = {w.strip(configs.Data.space_symbol) for w in vocab}
    vocab_no_space_symbol = {
        w: idx for idx, w in enumerate(vocab_no_space_symbol)
    }

    # keep track of which words are excluded - not a candidate for being inserted into test sentences
    nds = (configs.Dirs.external_words / "non-dictionary.txt").open().read().split()
    sws = (configs.Dirs.external_words / "stopwords.txt").open().read().split()
    excluded_words = set(nds + sws)

    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = lambda text: Doc(nlp.vocab, words=text.split())
    #nlp.tokenizer.add_special_case("<unk>", [{spacy.symbols.ORTH: "<unk>"}])

    print(f'Will count vocab words in the following corpora:')
    for c in args.corpora:
        print(c)

    # get information about all words in corpora
    corpus_names = [c.stem for c in args.corpora]
    df = pd.DataFrame(
        0,
        index=range(len(vocab_no_space_symbol)),
        columns=(
            POS_TAGS +
            ['total-frequency'] +
            [f'{corpus_name}-frequency' for corpus_name in corpus_names]
        )
    )
    for corpus_path, corpus_name in zip(args.corpora, corpus_names):
        with open(corpus_path) as f:
            sentences = list(map(str.lower, map(str.strip, f.readlines())))

        for n, sd in enumerate(nlp.pipe(sentences, disable=["parser", "ner"])):
            for sw in sd:
                update_row(
                    df, vocab_no_space_symbol[sw.text], sw.tag_, corpus_name
                )

            if args.dry_run and n+1 >= args.dry_run_n_sentences:
                break

            if n % 1000 == 0:
                print(f'{corpus_path.stem:<24} {n:>12,}/{len(sentences):>12,}')

    df.index: pd.core.indexes.base.Index = vocab_no_space_symbol.keys() # type: ignore
    df["is_excluded"] = df.index.map(is_excluded)

    not_occurring = df["total-frequency"] == 0
    tokens_not_in_corpus = df.index[not_occurring].to_list()
    print(f'Did not find {len(tokens_not_in_corpus)} tokens from vocab in corpora: {", ".join(tokens_not_in_corpus)}')
    df["is_excluded"] |= not_occurring

    df.sort_values(by='total-frequency', inplace=True, ascending=False)

    if args.dry_run:
        print(df)
        exit(f'Dry run completed after {args.dry_run_n_sentences} sentences')

    df.to_csv(configs.Dirs.data / 'vocab_words' / f'{vocab_name}.csv', index=True)
