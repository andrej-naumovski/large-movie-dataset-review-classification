from collections import Counter
from keras.preprocessing.text import text_to_word_sequence
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def filter_stop_words_from_text(text=''):
    word_array = text_to_word_sequence(text)
    new_text = " ".join([word for word in word_array if word not in ENGLISH_STOP_WORDS])

    return new_text


def count_repetitions_of_words_in_text(text=''):
    word_array = text_to_word_sequence(text)
    return Counter(word_array)


def map_text_to_index(text='', counter=Counter([])):
    word_array = text_to_word_sequence(text)
    indexed_array = [counter[word] for word in word_array]
    return indexed_array


def filter_words(word_counter, max_words):
    most_common = word_counter.most_common(max_words)
    return dict(most_common)
