import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors


def read_glove_file(embedding_dir,binary = False,embedding_type = 'glove'):
    filename = embedding_dir.split('/')[-1]

    if embedding_type == 'glove': 
        filename = '/'.join(embedding_dir.split('/')[:-1]+[f'gensim_{filename}'])
        glove2word2vec(glove_input_file = embedding_dir,
                    word2vec_output_file = filename)
    gensim_model = KeyedVectors.load_word2vec_format(filename, binary=binary)
    return gensim_model

def get_word_index_dicts(gensim_model):
    words = gensim_model.index2word
    i = 1
    word_to_index = {}
    index_to_words = {}
    for w in sorted(words):
        word_to_index[w] = i
        index_to_words[i] = w
        i = i + 1
    return word_to_index,index_to_words