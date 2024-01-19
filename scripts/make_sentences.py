"""
generate sentences for all paradigms.
"""
import argparse
from pathlib import Path
import importlib
from itertools import islice, chain

from zorro.vocab import get_vocab_words
from zorro.filter import collect_unique_pairs
from zorro.utils import get_phenomena_and_paradigms, capitalize_names_in_sentence
from zorro import configs


stop_words = (configs.Dirs.external_words / "stopwords.txt").open().read().split()
number_words = (configs.Dirs.legal_words / "number_words.txt").open().read().split()


def paired(a, n=2):
    a = iter(a)
    while True:
        p = []
        try:
            for i in range(n):
                p.append(next(a))
            yield tuple(p)
        except StopIteration:
            break


def sentence_pair_in_vocab(sentence_pair) -> bool:
    for sentence in sentence_pair:
        words_to_check = sentence.split()
        for w in words_to_check:
            if not(w in vocab_words or w.lower() in vocab_words or
                    w in stop_words or w.lower() in stop_words):
                return False
    return True


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--vocab_name", default=None
    )
    argparser.add_argument(
        "--out_path", type=Path, nargs="+",
        default=[Path("sentences")]
    )
    args = argparser.parse_args()

    vocab_words = get_vocab_words(args.vocab_name)

    paradigm_sentences = {}

    # for all paradigms
    for phenomenon, paradigm in get_phenomena_and_paradigms():

        print('***************************************************************************************')
        print(f'Making test sentences for {phenomenon} {paradigm} with vocab={configs.Data.vocab_name}')
        print('***************************************************************************************')

        paradigm_module = importlib.import_module(f'zorro.{phenomenon}.{paradigm}')

        # generate sentences once, in order to save the same sentences to two locations
        sentence_pairs = list(
            islice(
                collect_unique_pairs(
                    filter(
                        sentence_pair_in_vocab,
                        paired(paradigm_module.main()) # type: ignore
                    )
                ),
                configs.Data.num_pairs_per_paradigm
            )
        )

        sentences = list(chain.from_iterable(sentence_pairs))

        # capitalize proper nouns for case-sensitive models like roberta-base
        sentences = [capitalize_names_in_sentence(s) for s in sentences]

        # TODO save info about each sentence's template in the text file saved locally

        saved_sentences = []
        # check
        for sentence_pair in paired(sentences):
            if sentence_pair_in_vocab(sentence_pair):
                saved_sentences.extend(sentence_pair)
            elif False:
                print(sentence_pair[0])
                print(sentence_pair[1])

        print(f"Saving {len(saved_sentences)} sentences.")
        paradigm_sentences[(phenomenon, paradigm)] = saved_sentences

        if not saved_sentences:
            print('Did not generate any sentences.'
                  'This can occur if plural versions of singular nouns are not in vocab.')
            print(f'Skipping {paradigm}')
            continue

        # save each file in repository, and also on shared drive
        for out_path in [
            path / configs.Data.vocab_name / f'{phenomenon}-{paradigm}.txt'
            for path in args.out_path
        ]:
            print(f"save to path: {out_path}")
            if not out_path.parent.is_dir():
                out_path.parent.mkdir(parents=True)

            with open(out_path, 'w') as f:
                for sentence in saved_sentences:
                    # write to file
                    f.write(sentence + '\n')

            print(f'Saved sentences to {out_path}')

    print(f"n_total_saved_paradigms={len(paradigm_sentences)}")
    for (phenomenon, paradigm), sentences in paradigm_sentences.items():
        if len(sentences) < 4000:
            print(f"{phenomenon} {paradigm}: {len(sentences)}")