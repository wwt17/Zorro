"""
present words belonging to the same POS tag for human to judge as legal or not - can they be used in test sentences?
"""
import argparse
import pandas as pd

from zorro import configs
from zorro.vocab import load_vocab_df


tag2template = {
    'NN': 'look at this ADJ {}',
    'JJ': 'look at this {} NN',
    'VBD': 'sarah {} something',
    'VB': 'sarah might {} something',
    'VBG': 'sarah might be {} something',
    'VBZ': 'sarah {} something',
}
all_tags = list(tag2template.keys())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--vocab_name", default=None
    )
    argparser.add_argument(
        "--tag_max_freq", action="store_true",
        help="For each word in vocab, treat it as a legal word with its most tag."
    )
    argparser.add_argument(
        "--tags", nargs="+", choices=all_tags, default=all_tags
    )
    args = argparser.parse_args()

    vocab_df = load_vocab_df(args.vocab_name)

    if args.tag_max_freq:  #TODO: this ignores transitivity of verbs, and intransitive verbs may not fit into the templates
        assert set(args.tags) == set(all_tags), "All tags are processed with --tag_max_freq"
        my_tags_frequency = vocab_df[args.tags].sum(axis=1)
        vocab_df["other-tags"] = vocab_df["total-frequency"] - my_tags_frequency
        vocab_df["max-tag"] = vocab_df[args.tags + ["other-tags"]].idxmax(axis=1)
        not_excluded = vocab_df.loc[~vocab_df["is_excluded"], "max-tag"]
        for tag in args.tags:
            df_path = configs.Dirs.legal_words / f"{tag}.csv"
            df_legal = pd.DataFrame({
                "word": not_excluded[not_excluded == tag].index,
                "is_legal": 1,
            })
            df_legal.to_csv(df_path, index=False)

    else:
        for tag in args.tags:
            df_path = configs.Dirs.legal_words / f"{tag}.csv"
            if not df_path.exists():
                df_legal = pd.DataFrame(columns=['word'] + ['is_legal'])
            else:
                df_legal = pd.read_csv(df_path)

            # for each whole word in vocab, make new row for df
            for n, (vw, vw_series) in enumerate(vocab_df.iterrows()):
                if vw in df_legal['word'].tolist():
                    continue
                if vw_series['is_excluded']:
                    continue

                row = {'word': vw}

                # consult spacy tag if whole word can NOT be used in this slot
                if vw_series[tag] == 0:
                    row[f'is_legal'] = 0

                # ask user if whole word can be used in this slot
                else:
                    print()
                    print(tag2template[tag].format(f'\033[94m{vw}\033[0m'))   # uses color

                    is_valid = False
                    while not is_valid:
                        response = input('Grammatical? [f=yes j=no q=quit]')
                        if response == 'f':
                            row[f'is_legal'] = 1
                            is_valid = True
                        elif response == 'j':
                            row[f'is_legal'] = 0
                            is_valid = True
                        elif response == 'q':
                            exit('User exit')
                        else:
                            is_valid = False

                df_legal = pd.concat(
                    [df_legal, pd.DataFrame.from_records([row])],
                    ignore_index=True
                )
                df_legal.to_csv(df_path, index=False)
                print(row)
                print(f'\nSaved {n}/{len(vocab_df)}\n')