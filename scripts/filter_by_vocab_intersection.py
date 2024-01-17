"""
present words belonging to the same POS tag for human to judge as legal or not - can they be used in test sentences?
"""
import argparse
import json
from pathlib import Path
import pandas as pd

from zorro import configs
from zorro.vocab import load_vocab_df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--vocab_file", type=Path, nargs="*", default=[],
        help="Path(s) to the vocab file(s). Used to filter out OOV words."
    )
    argparser.add_argument(
        "--vocab_name", default=None
    )
    argparser.add_argument(
        "--out_vocab_file", required=True
    )
    args = argparser.parse_args()

    vocabs = []
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
        vocabs.append(vocab)

    intersect_vocab = None
    for vocab in vocabs:
        if intersect_vocab is None:
            intersect_vocab = set(vocab.keys())
        else:
            intersect_vocab &= set(vocab.keys())
    print(intersect_vocab)

    vocab_df = load_vocab_df(args.vocab_name, return_excluded_words=True)
    not_in_intersection = vocab_df.index.map(lambda word: word not in intersect_vocab) # type: ignore

    filtered_vocab_df = vocab_df[~not_in_intersection]
    filtered_vocab_df.to_csv(args.out_vocab_file)