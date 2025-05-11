# WRD implementaion for Japanese texts
This repository is a WRD(Word Rotator's Distance) implemenation for Japanese texts.
Slightly faster than the original codes.
WRD is an improved algorithm of WMD(Word Mover's Distance) and achieves great performances in measuring textual similarities.
See the details in the [paper](https://arxiv.org/abs/2004.15003v1).

## Requirements
### Install Python modules
- [gensim](https://pypi.org/project/gensim/)
- [mecab-python3](https://pypi.org/project/mecab-python3/) with [MeCab](https://taku910.github.io/mecab/) and [Neologd dictionary](https://github.com/neologd/mecab-ipadic-neologd)
- [POT](https://pythonot.github.io/)

### Download pre-trained vectors for Word2vec
From [this site](https://github.com/singletongue/WikiEntVec/releases), download pre-trained vectors trained on Japanese Wikipedia and set it in `vecs` directory.
`jawiki.word_vectors.200d.txt.bz2` is recommended.

If you want, your original pre-trained vectors could be used.

## Usage
Just run the below command. `--mecab_dict_path` is an optional argument.

```
$ python main.py --mecab_dict_path /usr/local/lib/mecab/dic/mecab-ipadic-neologd/
```

If successful, the result will be as follows.
The value of the first line is the time taken to execute the pathfinders's code.
The value of the final line is the time it takes to execute my code.
My code is about 5 times faster than the pathfinder's code.

```
11.4994708340
1.9490371520
```

If you'd like to change the sentences to compare, edit the [sentences.txt](/sentences.txt) directly. 

## Thanks
the utmost gratitude to kenta1984 for his initial work.
