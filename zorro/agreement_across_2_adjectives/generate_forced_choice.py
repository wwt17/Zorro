
from zorro.agreement_across_2_adjectives.shared import task_name, pre_nominals_singular, pre_nominals_plural, plural
from zorro.task_words import get_task_word_combo
from zorro.vocab import get_vocab_words

NUM_ADJECTIVES = 2
NUM_NOUNS = 7

template1 = 'look at {} {} {} {} .'
template2 = '{} {} {} {} went there .'

rules = {
    ('JJ', 0, NUM_ADJECTIVES): [
        template1.format('this', '{}', '_', '_'),
        template2.format('this', '{}', '_', '_'),
    ],
    ('JJ', 1, NUM_ADJECTIVES): [
        template1.format('this', '_', '{}', '_'),
        template2.format('this', '_', '{}', '_'),
    ],
    ('NN', 0, NUM_NOUNS): [
        template1.format('this', '_', '_', '{}'),
        template2.format('this', '_', '_', '{}'),
    ],
}


def main():
    """
    example:
    "look at this green red house ." vs. "look at this green red houses ."
    "this green red house went there ." vs. "this green red houses went there."
    """

    noun_plurals = get_vocab_words(tag='NNS')

    for pre_nominal in pre_nominals_singular + pre_nominals_plural:

        for words_singular in get_task_word_combo(task_name, rules.keys()):
            noun_plural = plural.plural(words_singular[2])
            if noun_plural not in noun_plurals:
                continue
            words_plural = [words_singular[0], words_singular[1], noun_plural]

            yield template1.format(pre_nominal, *words_singular)
            yield template1.format(pre_nominal, *words_plural)

            yield template2.format(pre_nominal, *words_singular)
            yield template2.format(pre_nominal, *words_plural)


if __name__ == '__main__':
    for n, s in enumerate(main()):
        print(f'{n:>12,}', s)