"""
make csv file (e.g. vocab_words/babyberta.csv) that contains tokens in a tokenizer configuration file,
alongside their frequency in corpora of interest (e.g. childes, newsela, wikipedia).

A huggingface tokenizers v0.10 configuration file is expected.
"""
from typing import Optional
import argparse
import spacy
from spacy.tokens import Doc
import json
from pathlib import Path
import pandas as pd
import tqdm
from zorro import configs


POS_TAGS = [
    'CD'  ,  # cardinal number
    'NN'  ,
    'NNS' ,
    'NNP' ,  # proper noun
    'NNPS',
    'VB'  ,  # base form of verb
    'VBD' ,  # verb past tense
    'VBG' ,  # verb gerund or present participle
    'VBN' ,  # verb past participle
    'VBP' ,  # verb non-3rd person singular present
    'VBZ' ,  # verb 3rd person singular present
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
        tag: Optional[str],
        corpus_name: str,
):
    if tag is not None:
        if tag not in df.columns:
            df[tag] = 0
        df.loc[idx, tag] += 1 # type: ignore
    df.loc[idx, 'total-frequency'] += 1 # type: ignore
    df.loc[idx, f'{corpus_name}-frequency'] += 1 # type: ignore


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--out_name", default="vocab",
        help="The name of the output vocab file."
    )
    argparser.add_argument(
        "--vocab_file", type=Path, nargs="*", default=[],
        help="Path(s) to the vocab file(s)."
    )
    argparser.add_argument(
        "--corpora", type=Path, nargs="*", required=True,
        help="Paths to the corpora."
    )
    argparser.add_argument(
        "--skip_pos_tag_corpora", default=["childes"],
        help="Skip getting POS tag frequencies in these corpora (given by their names)."
    )
    argparser.add_argument(
        "--whole_word", action="store_true",
        help="Ensure remaining words to be whole words."
    )
    argparser.add_argument(
        "--exclude_not_occurring_words", choices=["all", "any"], default="any",
        help="Exclude words that do not occur in (any|all) corpora."
    )
    argparser.add_argument(
        "--dry_run", action="store_true",
    )
    argparser.add_argument(
        "--dry_run_n_sentences", type=int, default=100
    )
    args = argparser.parse_args()

    corpora = args.corpora
    corpus_names = []
    args.corpora = []
    for corpus_path in corpora:
        if corpus_path.is_dir():
            corpus_name = corpus_path.name
            args.vocab_file.append(corpus_path/"vocab.json")
            args.corpora.append(corpus_path/"train.txt")
        else:
            corpus_name = corpus_path.stem
            args.corpora.append(corpus_path)
        corpus_names.append(corpus_name)
    print(f'Will count vocab words in the following corpora:')
    for c in args.corpora:
        print(c)

    merged_vocab = set()
    for vocab_file in args.vocab_file:
        vocab_name = vocab_file.stem
        with open(vocab_file) as f:
            vocab = json.load(f)
        try:
            vocab = vocab['model']['vocab']  # get vocab from tokenizer, without space symbol
        except:
            pass
        assert isinstance(vocab, dict)
        assert max(vocab.values()) + 1 == len(vocab)
        merged_vocab |= vocab.keys()
    vocab = {word: idx for idx, word in enumerate(merged_vocab)}
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

    # get information about all words in corpora
    df = pd.DataFrame(
        0,
        index=range(len(vocab_no_space_symbol)),
        columns=(
            ['total-frequency'] +
            [f'{corpus_name}-frequency' for corpus_name in corpus_names] +
            POS_TAGS
        )
    )
    for corpus_path, corpus_name in zip(args.corpora, corpus_names):
        with open(corpus_path) as f:
            sentences = list(map(str.lower, map(str.strip, f.readlines())))

        if corpus_name in args.skip_pos_tag_corpora:
            for sentence in tqdm.tqdm(sentences, desc=corpus_name):
                words = sentence.split()
                for word in words:
                    update_row(
                        df, vocab_no_space_symbol[word], None, corpus_name
                    )

        else:
            for n, sd in enumerate(nlp.pipe(tqdm.tqdm(sentences, desc=corpus_name), disable=["parser", "attribute_ruler", "lemmatizer", "ner"])):
                for sw in sd:
                    update_row(
                        df, vocab_no_space_symbol[sw.text], sw.tag_, corpus_name
                    )

                if args.dry_run and n+1 >= args.dry_run_n_sentences:
                    break

    df.index: pd.core.indexes.base.Index = vocab_no_space_symbol.keys() # type: ignore
    df["is_excluded"] = df.index.map(is_excluded)

    not_occurring_in_all = df["total-frequency"] == 0
    tokens_not_in_all_corpora = df.index[not_occurring_in_all].to_list()
    print(f'Did not find {len(tokens_not_in_all_corpora)} tokens from vocab in all corpora: {", ".join(tokens_not_in_all_corpora)}')

    not_occurring_in_any = False
    for corpus_name in corpus_names:
        not_occurring_in_any |= df[f"{corpus_name}-frequency"] == 0
    tokens_not_in_any_corpus = df.index[not_occurring_in_any].to_list()
    print(f'Did not find {len(tokens_not_in_any_corpus)} tokens from vocab in any corpus: {", ".join(tokens_not_in_any_corpus)}')

    df["is_excluded"] |= {
        "all": not_occurring_in_all,
        "any": not_occurring_in_any
    }[args.exclude_not_occurring_words]

    df.sort_values(by='total-frequency', inplace=True, ascending=False)

    if args.dry_run:
        print(df)
        exit(f'Dry run completed after {args.dry_run_n_sentences} sentences')

    df.to_csv(configs.Dirs.data / 'vocab_words' / f'{args.out_name}.csv', index=True)
