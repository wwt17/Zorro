import random

NUM_NOUNS = 100
NUM_ADJECTIVES = 50

template1 = 'the {} on the {} {} {} .'
template2 = 'the {} by the {} {} {} .'

rules = {
    ('NN', 0, NUM_NOUNS): [
        template1.format('{}', '_', 'is', '_'),
        template2.format('{}', '_', 'is', '_'),
    ],
    ('NN', 1, NUM_NOUNS): [
        template1.format('_', '{}', 'is', '_'),
        template2.format('_', '{}', 'is', '_'),
    ],
    ('JJ', 0, NUM_ADJECTIVES): [
        template1.format('_', '_', 'is/are', '{}'),
        template2.format('_', '_', 'is/are', '{}'),
    ],

}


def main():
    """
    example:
    "the dog on the mats is brown" vs "the dog on the mats are brown"

    considerations:
    1. use equal proportion of sentences containing plural vs. singular subject nouns
    2. use equal proportion of sentences containing plural vs. singular object nouns
    2. subject with object number is counterbalanced such that:
        -singular subjects occur with 50:50 singular:plural objects
        -plural   subjects occur with 50:50 singular:plural objects
    """

    from zorro.task_words import get_task_words
    from zorro.agreement_across_PP.shared import paradigm, plural, copulas_singular, copulas_plural
    from zorro.vocab import get_vocab_words
    from zorro import configs

    noun_plurals = get_vocab_words(tag='NNS')
    subjects_s = get_task_words(paradigm, tag='NN', order=0)
    objects_s = get_task_words(paradigm, tag='NN', order=1)
    adjectives = get_task_words(paradigm, tag='JJ')

    num_pairs = 0

    while num_pairs < configs.Data.num_pairs_per_paradigm:

        # TODO duplicate combinations are not excluded - do not sample with replacement

        # counter-balance singular vs plural with subj vs. obj
        sub_s = random.choice(objects_s)
        obj_s = random.choice(subjects_s)
        sub_p = plural.plural(sub_s)
        obj_p = plural.plural(obj_s)
        if sub_p not in noun_plurals or obj_p not in noun_plurals:
            continue
        if sub_s == sub_p or obj_s == obj_p:  # exclude nouns with ambiguous number
            continue

        # random choices
        template = random.choice([template1, template2])
        copula = random.choice(copulas_singular + copulas_plural)
        adj = random.choice(adjectives)

        # contrast is in number agreement between subject and copula
        yield template.format(sub_s, obj_s, copula, adj)
        yield template.format(sub_p, obj_s, copula, adj)

        # same as above, except that object number is opposite
        yield template.format(sub_s, obj_p, copula, adj)
        yield template.format(sub_p, obj_p, copula, adj)

        num_pairs += 2


if __name__ == '__main__':
    for n, s in enumerate(main()):
        print(f'{n//2:>12,}', s)