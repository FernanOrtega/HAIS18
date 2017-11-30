import sys
import numpy as np
from datetime import time
from random import shuffle

from keras.preprocessing import sequence
from sklearn.model_selection import KFold

from word_vectorizer import WordEmbeddings


def candidates_of_node(node):
    sequence = [(node[0], node[1])]
    l_result = []
    if len(node) > 2:
        for i in range(2, len(node)):
            c_sequence, c_l_result = candidates_of_node(node[i])
            sequence += c_sequence
            l_result.extend(c_l_result)

    l_result.append(sequence)

    return sequence, l_result


def max_hits(n):
    # return (n + n**2)/2
    return sum([score_funct(i + 1) for i in range(n)])


def score_funct(x):
    # return 0.5+ 1 / np.math.sqrt(2 * x)
    # return 1 / np.math.sqrt(x)
    # return max(1-0.05*x, 0)
    return 0 if x <= 0 else 1 / x


def one_hot_encode(number, size_vector):
    encoded = np.zeros(size_vector)
    if number >= size_vector:
        raise ValueError('number must be less than size_vector')
    encoded[number] = 1

    return encoded


def one_hot_decode(encoded_number):
    return np.argmax(encoded_number)


def compute_candidates(row, w2v_model):
    l_cand_of_deptree = []
    tokens = row[0]
    deptree = row[1]
    conditions = row[2]
    for sequence in candidates_of_node(deptree)[1]:
        if len(sequence) > 1:
            sequence.sort(key=lambda x: x[0])
            l_index_sequence = [w_index for (w_index, dep_idx) in sequence]
            set_index_sequence = set(l_index_sequence)
            tokens_sequence = [(w2v_model.word2idx(tokens[i - 1]), i_dep) for (i, i_dep) in
                               sequence]

            if len(conditions) > 0:
                array_hits = [sum([value * score_funct(index + 1) for index, value in
                                   enumerate([int(i in set_index_sequence) for i in cond])])
                              for cond in conditions]
                score = max([2.0 * array_hits[index] / (2.0 * array_hits[index] +
                                                        (max_hits(len(condition)) + max_hits(len(sequence)) - 2.0 *
                                                         array_hits[index]))
                             for index, condition in enumerate(conditions)])
            else:
                score = 0.0

            l_cand_of_deptree.append([l_index_sequence, tokens_sequence, score])

    return l_cand_of_deptree


# TODO implement it
def fit_model(train, w2v_model):
    pass


# TODO implement it
def evaluate(test, model):
    pass


# TODO implement it
def save_results(results):
    pass


def main(dataset, w2v_model, n_splits, output_csv_path):
    results = []

    shuffle(dataset)
    # train/test splits or k-fold
    folds = KFold(n_splits=n_splits, random_state=7, shuffle=False)
    splits = [(train_index, test_index) for train_index, test_index in folds.split(dataset)]

    prep_dataset = [[row[0], row[1], row[2], compute_candidates(row, w2v_model)] for row in dataset]
    for k, (train_index, test_index) in enumerate(splits):
        train, test = prep_dataset[train_index], prep_dataset[test_index]
        # TODO for each model to test
        model = fit_model(train, w2v_model)
        results.append(evaluate(test, model))

    save_results(results)


if __name__ == '__main__':

    args = sys.argv[1:]

    if len(args) != 4:
        print('Wrong number of arguments')
        print('Usage (relative paths!!): main.py <dataset path> <lang> <word2vec model path>'
              '<# folds> <output csv path>')
        exit()

    dataset = [eval(line) for line in open(args[0], encoding='utf-8')]
    lang = args[1]
    start = time.time()
    w2v_model = WordEmbeddings(lang, args[2])
    end = time.time()
    print('WV loaded', (end - start))
    n_splits = int(args[3])
    output_csv_path = args[4]

    main(dataset, w2v_model, n_splits, output_csv_path)
