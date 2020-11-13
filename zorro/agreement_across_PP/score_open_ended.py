from typing import List, Dict

from zorro.agreement_across_PP.shared import templates, copulas_singular, copulas_plural
from zorro.agreement_across_PP.shared import nouns_plural, nouns_singular
from zorro import configs

prediction_categories = (
    's',
    "correct\ncopula",
    "false\ncopula",
    "other")


def categorize_by_template(sentences_in, sentences_out: List[List[str]]):

    template2sentences_out = {}
    template2mask_index = {}

    for s1, s2 in zip(sentences_in, sentences_out):
        template2sentences_out.setdefault(templates[0], []).append(s2)
        if templates[0] not in template2mask_index:
            template2mask_index[templates[0]] = s1.index(configs.Data.mask_symbol)
    return template2sentences_out, template2mask_index


def categorize_predictions(sentences_out: List[List[str]],
                           mask_index: int) -> Dict[str, float]:

    res = {k: 0 for k in prediction_categories}

    for sentence in sentences_out:
        predicted_word = sentence[mask_index]  # predicted word may not be a copula ("is", "are")
        targeted_noun = sentence[1]

        if predicted_word == 's':
            res['s'] += 1

        elif targeted_noun in nouns_plural and predicted_word in copulas_plural:
            res["correct\ncopula"] += 1

        elif targeted_noun in nouns_singular and predicted_word in copulas_singular:
            res["correct\ncopula"] += 1

        elif targeted_noun in nouns_plural and predicted_word in copulas_singular:
            res["false\ncopula"] += 1

        elif targeted_noun in nouns_singular and predicted_word in copulas_plural:
            res["false\ncopula"] += 1

        else:
            res['other'] += 1

    return res