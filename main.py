import os
import argparse
import itertools
import time

from gensim.models import KeyedVectors
import MeCab

import pathfinders_main
import followers_main

def measure_time(calc_similarity, sentence_vecs, combinations):
    def calc_similarity_wrapper(combination):
        return calc_similarity(sentence_vecs, combination)

    start = time.perf_counter()
    result = list(map(calc_similarity_wrapper, combinations))
    end = time.perf_counter()
    
    print('{:.10f}'.format((end - start)))
    # print(result)

def main(args):
    mt = MeCab.Tagger('-d {} -Owakati'.format(args.mecab_dict_path)) if args.mecab_dict_path is not None else MeCab.Tagger('-Owakati')
    wv = KeyedVectors.load_word2vec_format(os.path.dirname(os.path.abspath(__file__)) + '/vecs/jawiki.word_vectors.200d.txt')

    with open("sentences.txt", "r") as f:
        sentences = f.readlines()

    def get_w_wrapper(sentence):
        return pathfinders_main.get_w(sentence, mt, wv)

    sentence_vecs = list(map(get_w_wrapper, sentences))
    combinations = list(itertools.combinations(range(len(sentence_vecs)), 2))

    measure_time(pathfinders_main.calc_similarity, sentence_vecs, combinations)
    measure_time(followers_main.calc_similarity, sentence_vecs, combinations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mecab_dict_path', type=str,
        help='Path to MeCab custom dictionary.')
    args = parser.parse_args()

    main(args)