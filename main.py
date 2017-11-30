import sys
from random import shuffle

from word_vectorizer import WordEmbeddings


def main(dataset, w2v_model, k, output_csv_path):
    shuffle(dataset)


if __name__ == '__main__':

    args = sys.argv[1:]

    if len(args) != 4:
        print('Wrong number of arguments')
        print('Usage (relative paths!!): main.py <dataset path> <lang> <word2vec model path>'
              '<# folds> <output csv path>')
        exit()

    dataset = [eval(line) for line in open(args[0], encoding='utf-8')]
    lang = args[1]
    w2v_model = WordEmbeddings(lang, args[2])
    k = int(args[3])
    output_csv_path = args[4]

    main(dataset, w2v_model, k, output_csv_path)