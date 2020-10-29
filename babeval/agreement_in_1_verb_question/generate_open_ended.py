import random

from babeval.agreement_in_1_verb_question import *

NUM_NOUNS_FROM_EACH_LIST = 400  # there are only 414 plurals

template1 = 'where' + f' {configs.Data.mask_symbol} ' + 'the {} ?'
template2 = 'what' + f' {configs.Data.mask_symbol} ' + 'the {} ?'


def main():
    random.seed(configs.Data.seed)

    nouns_balanced = random.sample(nouns_singular, k=NUM_NOUNS_FROM_EACH_LIST) + \
                     random.sample(nouns_plural, k=NUM_NOUNS_FROM_EACH_LIST)

    for noun in nouns_balanced:
        yield template1.format(noun)
        yield template2.format(noun)

