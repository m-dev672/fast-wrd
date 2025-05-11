import os
import argparse
import itertools
import time

from gensim.models import KeyedVectors
import MeCab
import numpy as np
import ot

import pathfinders_main

def calc_similarity(sentence_vecs, combination):
    w1 = sentence_vecs[combination[0]]
    w2 = sentence_vecs[combination[1]]

    w1_norm = np.linalg.norm(w1, axis=1)
    m1 = w1_norm / w1_norm.sum()

    w2_norm = np.linalg.norm(w2, axis=1)
    m2 = w2_norm / sum(w2_norm)

    # Compute cost matrix C
    w_dot = np.dot(w1, w2.T)
    w_norm = np.outer(w1_norm, w2_norm.T)
    c = 1 - w_dot / w_norm

    # Show the result
    return ot.emd2(m1, m2, c)

def main(args):
    mt = MeCab.Tagger('-d {} -Owakati'.format(args.mecab_dict_path)) if args.mecab_dict_path is not None else MeCab.Tagger('-Owakati')
    wv = KeyedVectors.load_word2vec_format(os.path.dirname(os.path.abspath(__file__)) + '/vecs/jawiki.word_vectors.200d.txt')

    with open("sentences.txt", "r") as f:
        sentences = f.readlines()

    def get_w_wrapper(sentence):
        return pathfinders_main.get_w(sentence, mt, wv)

    sentence_vecs = list(map(get_w_wrapper, sentences))
    combinations = list(itertools.combinations(range(len(sentence_vecs)), 2))

    def calc_similarity_wrapper(combination):
        return calc_similarity(sentence_vecs, combination)

    start = time.perf_counter()
    result = list(map(calc_similarity_wrapper, combinations))
    end = time.perf_counter()
    
    print(end - start)
    # print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mecab_dict_path', type=str,
        help='Path to MeCab custom dictionary.')
    args = parser.parse_args()

    main(args)