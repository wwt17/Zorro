import random

from zorro.filter import collect_unique_pairs
from zorro.words import get_legal_words
from zorro.counterbalance import find_counterbalanced_subset
from zorro import configs

template1 = 'no {} {} {} {} {} {} .'


def main():
    """
    example:
    "no cat can jump on more than two dogs ." vs. "no cat jump on at least two dogs ."
    """

    nouns_s_and_p = get_legal_words(tag='NN', second_tag='NNP')
    number_words_ = (configs.Dirs.legal_words / "number_words.txt").open().read().split()
    number_words = find_counterbalanced_subset(number_words_,min_size=6, max_size=len(number_words_))

    quantifiers_g_b = [('more than', 'at least'),
                       #('fewer than', 'at most'),
                       ]

    animates_ = (configs.Dirs.legal_words / 'animates.txt').open().read().split()
    animates = find_counterbalanced_subset(animates_, min_size=8, max_size=len(animates_))

    # a linker can be a preposition or determiner phrase
    verb2linker = {
        #'become': None,
        'catch': None,
        'leave': None,
        #'increase': 'the size of',
        'move': 'to',
        'open': 'the door to',
        #'exist': 'without',
        #'contain': None,
        'stand': 'on top of',
        'change': None,
        #'surround': None,
        'carry': None,
        #'act': 'like',
        'see': None,
        'look': 'at',
        'say': None,
        'tell': None,
        'play': None,
        'make': None,
        'take': None,
        'give': None,
        'find': None,
        'show': None,
        'talk': 'about',
        'pick': None,
        'hit': None,
    }
    verbs_ = list(verb2linker.keys())
    verbs = find_counterbalanced_subset(verbs_, min_size=len(verbs_), max_size=len(verbs_))
    print(f"#verbs: {len(verbs)}")

    while True:

        # random choices
        animate = random.choice(animates)
        noun_s, noun_p = random.choice(nouns_s_and_p)
        number_word = random.choice(number_words)
        quantifier_g, quantifier_b = random.choice(quantifiers_g_b)
        verb = random.choice(verbs)
        aux = random.choice(['can', 'could'])

        verb_and_optional_linker = verb
        if verb2linker[verb] is not None:
            verb_and_optional_linker += ' ' + verb2linker[verb]

        if number_word == 'one':
            noun = noun_s
        else:
            noun = noun_p

        yield template1.format(animate, aux, verb_and_optional_linker, quantifier_b, number_word, noun)  # bad
        yield template1.format(animate, aux, verb_and_optional_linker, quantifier_g, number_word, noun)  # good


if __name__ == '__main__':
    for n, s in enumerate(collect_unique_pairs(main)):
        print(f'{n//2+1:>12,}', s)
