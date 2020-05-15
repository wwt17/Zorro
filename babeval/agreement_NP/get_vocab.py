# find 4096 most frequent words from the master_word_list

from collections import Counter
import nltk

with open('childes-20191206_vocab.txt') as f:
    text_string_list = []
    text_string = f.read().split()
    text_string_list.append(text_string)
    my_list = [t[1::2] for t in text_string_list]
    for lst in my_list:
        clean_word_list = lst[:4096]
        for word in clean_word_list:
          print(word, file = open("childes-20191206_mlm_4096.txt","a"))

# get pos_tag
list_of_strings = " ".join(clean_word_list)
text = nltk.word_tokenize(list_of_strings)
POS_list = nltk.pos_tag(text)

#separate singular and plural nouns
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
def isplural(word):
    lemma = wnl.lemmatize(word, 'n')
    plural = True if word is not lemma else False
    return plural, lemma
def merge(list1, list2): 
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
    return merged_list 

file_name_1 = "nouns.txt"
input_directory = "word_lists/"

with open(input_directory + file_name_1) as f:
    words_list = f.read().split("\n")
    word_list = []
    isp_list = []
    for word in words_list:
        isp, lemma = isplural(word)
        word_list.append(word)
        isp_list.append(isp)
    merge_list = merge(words_list,isp_list)
    all_plural = [i[0] for i in merge_list if i[1] is False]
    all_singular = [i[0] for i in merge_list if i[1] is True]




