from typing import Callable, Generator, Iterable

from zorro import configs


def collect_unique_pairs(
        sentence_pairs: Iterable[tuple[str, str]],
) -> Generator[tuple[str, str], None, None]:
    """
    given a generator of sentence pairs, yield only consecutive sentence pairs if the pair was not previously collected
    """
    sentences1 = set()
    sentences2 = set()
    for sentence1, sentence2 in sentence_pairs:
        if sentence1 == sentence2:
            print(sentence1)
            print(sentence2)
            raise RuntimeError('Found pair of identical sentences')

        if sentence2 not in sentences2:  # check if good/grammatical sentence was not previously collected
            yield (sentence1, sentence2)
        sentences1.add(sentence1)
        sentences2.add(sentence2)