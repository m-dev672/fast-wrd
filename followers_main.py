import os
import argparse
import itertools

from gensim.models import KeyedVectors
import MeCab
import numpy as np
import ot


def get_w(text, mt, wv):
    kws = mt.parse(text).split()
    w = np.array([np.array(wv[kw]) for kw in kws if kw in wv])
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