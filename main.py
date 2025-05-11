import os
import argparse
import itertools

from gensim.models import KeyedVectors
import MeCab
import numpy as np
import ot


def get_w(text, mt, wv):
    kws = mt.parse(text).split()
    w = [np.array(wv[kw]) for kw in kws if kw in wv]
    return w


def get_z(w):
    z = 0
    for w_i in w:
        z += np.linalg.norm(w_i)
    return z


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def calc_similarity(sentence_vecs, combination):
    w1 = sentence_vecs[combination[0]]
    w2 = sentence_vecs[combination[1]]

    z1 = get_z(w1)
    z2 = get_z(w2)

    m1 = [np.linalg.norm(w1_i) / z1 for w1_i in w1]
    m2 = [np.linalg.norm(w2_i) / z2 for w2_i in w2]

    # Compute cost matrix C
    c = []
    for w1_i in w1:
        c.append([1 - cos_sim(np.array(w1_i), np.array(w2_j)) for w2_j in w2])

    # Show the result
    return ot.emd2(m1, m2, c)

def main(args):
    mt = MeCab.Tagger('-d {} -Owakati'.format(args.mecab_dict_path)) if args.mecab_dict_path is not None else MeCab.Tagger('-Owakati')
    wv = KeyedVectors.load_word2vec_format(os.path.dirname(os.path.abspath(__file__)) + '/vecs/jawiki.word_vectors.200d.txt')

    with open("sentences.txt", "r") as f:
        sentences = f.readlines()

    def get_w_wrapper(sentence):
        return get_w(sentence, mt, wv)

    sentence_vecs = list(map(get_w_wrapper, sentences))
    combinations = list(itertools.combinations(range(len(sentence_vecs)), 2))

    def calc_similarity_wrapper(combination):
        return calc_similarity(sentence_vecs, combination)

    print(list(map(calc_similarity_wrapper, combinations)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mecab_dict_path', type=str,
        help='Path to MeCab custom dictionary.')
    args = parser.parse_args()

    main(args)