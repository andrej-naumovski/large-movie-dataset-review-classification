import preprocessing
import pandas as pd
from os import listdir


def count_tokenize_dataset(dataset_text, max_words):
    filtered_text = preprocessing.filter_stop_words_from_text(dataset_text)
    word_counter = preprocessing.count_repetitions_of_words_in_text(filtered_text)
    return preprocessing.filter_words(word_counter, max_words)


def load_dataset(max_words):
    pos_train_content = append_files_content_to_string('./dataset/train/pos')
    neg_train_content = append_files_content_to_string('./dataset/train/neg')
    pos_test_content = append_files_content_to_string('./dataset/test/pos')
    neg_test_content = append_files_content_to_string('./dataset/test/neg')

    total_content = '{}{}{}{}'.format(pos_train_content, neg_train_content, pos_test_content, neg_test_content)

    tokenized_words = count_tokenize_dataset(total_content, max_words)

    X_train, y_train = load_reviews('train', './dataset', tokenized_words)

    X_test, y_test = load_reviews('test', './dataset', tokenized_words)

    return (X_train, y_train), (X_test, y_test)


def load_reviews(type, basepath, dictionary):
    pos_reviews_path = '{}/{}/{}'.format(basepath, type, 'pos')
    neg_reviews_path = '{}/{}/{}'.format(basepath, type, 'neg')
    X_pos = []

    X_neg = []

    pos_review_filenames = listdir(pos_reviews_path)
    for filename in pos_review_filenames:
        file = open(filename, 'r', encoding='utf-8')
        content = file.read()
        X_pos.append(content)
        file.close()

    X_pos = map(lambda x: preprocessing.map_text_to_index(x, dictionary), X_pos)

    y_pos = [1] * len(X_pos)

    neg_review_filenames = listdir(neg_reviews_path)
    for filename in neg_review_filenames:
        file = open(filename, 'r', encoding='utf-8')
        content = file.read()
        X_neg.append(content)
        file.close()

    y_neg = [0] * len(X_neg)

    X = X_pos + X_neg
    y = y_pos + y_neg

    return X, y


def append_files_content_to_string(basepath):
    file_list = listdir(basepath)
    content = ''
    for filename in file_list:
        file = open('{}/{}'.format(basepath, filename), 'r', encoding='utf-8')
        file_content = file.read()
        content = '{}{} '.format(content, file_content.replace('<br /><br />', ' '))
        file.close()

    return content
