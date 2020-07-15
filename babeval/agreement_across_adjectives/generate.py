from pathlib import Path
import random

from babeval.vocab import get_vocab

template = 'look at {} {} [MASK] .'

start_words = ['this', 'these', 'that', 'those']

adjectives_list = (Path(__file__).parent / 'word_lists' / 'adjectives_annotator1.txt').open().read().split()
adjectives_list = [w for w in adjectives_list if w in get_vocab()]


def main():
    """
    use adjectives specifically selected for this task
    """

    random.seed(2)

    for start_word in start_words:

        al1 = random.sample(adjectives_list, k=len(adjectives_list))
        al2 = random.sample(adjectives_list, k=len(adjectives_list))
        al3 = random.sample(adjectives_list, k=len(adjectives_list))

        for adj1, adj2, adj3 in zip(al1, al2, al3):
            yield template.format(start_word, ' '.join([adj1]))
            yield template.format(start_word, ' '.join([adj1, adj2]))
            yield template.format(start_word, ' '.join([adj1, adj2, adj3]))