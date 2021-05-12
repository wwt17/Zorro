import random

NUM_ADJECTIVES = 50
NUM_NOUNS = 90

template = '{} {} {} {} {} .'


def main():
    """
    example:
    "a big dog fell just now ." vs. "a big dog fallen just now ."

    """

    from zorro.irregular_verb_intransitive.shared import paradigm, determiners
    from zorro.irregular_verb_intransitive.shared import vb2vbd_vbn_intransitive
    from zorro.task_words import get_task_words
    from zorro.vocab import get_vocab_words
    from zorro import configs

    vocab = get_vocab_words()
    adjectives = get_task_words(paradigm, 'JJ', 0, NUM_ADJECTIVES)
    modifiers = ['just now', 'over there', 'some time ago', 'without us']
    nouns_s = get_task_words(paradigm, 'NN', 0, NUM_NOUNS)
    verbs = list(vb2vbd_vbn_intransitive.keys())

    def gen_sentences():
        while True:

            # random choices
            noun = random.choice(nouns_s)
            verb = random.choice(verbs)  # not counterbalanced across corpora
            det = random.choice(determiners)
            adj = random.choice(adjectives)
            mod = random.choice(modifiers)

            # get two contrasting irregular inflected forms
            vbd, vbn = vb2vbd_vbn_intransitive[verb]  # past, past participle
            if (vbd not in vocab or vbn not in vocab) or vbd == vbn:
                # print(f'"{verb_base:<22} excluded due to some forms not in vocab')
                continue

            # vbd is correct
            yield template.format(det, adj, noun, vbd, mod)
            yield template.format(det, adj, noun, vbn, mod)

            # vbn is correct
            yield template.format(det, adj, noun, 'had ' + vbd, mod)
            yield template.format(det, adj, noun, 'had ' + vbn, mod)

    # only collect unique sentences
    sentences = set()
    gen = gen_sentences()
    while len(sentences) // 2 < configs.Data.num_pairs_per_paradigm:
        sentence = next(gen)
        if sentence not in sentences:
            yield sentence
        sentences.add(sentence)


if __name__ == '__main__':
    for n, s in enumerate(main()):
        print(f'{n//2+1:>12,}', s)