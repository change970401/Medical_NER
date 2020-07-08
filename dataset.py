# -*- coding: utf-8 -*-
from util import CONSTANTS
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

'''构造数据集'''
def build_data(data_path):
    datas = []
    sample_x = []
    sample_y = []
    for line in open(data_path, 'r', encoding='utf-8'):
        line = line.rstrip().split('\t')
        if not line:
            continue
        char = line[0]
        if not char:
            continue
        cate = line[-1]
        if char is not '.':
            sample_x.append(char)
            sample_y.append(cate)
        if char in ['。', '?', '!', '！', '？', '.']:
            datas.append([sample_x, sample_y])
            sample_x = []
            sample_y = []
    return datas

'''将数据转换成keras所需的格式'''
def modify_data(data_path):
    datas = build_data(data_path)
    with open(CONSTANTS[4], 'rb') as f:
        word_dictionary = pickle.load(f)
    with open(CONSTANTS[5], 'rb') as f:
        label_dictionary = pickle.load(f)
    # vocab_size = len(word_dictionary.keys())
    # label_size = len(label_dictionary.keys())
    TIME_STAMPS = 300
    x = [[word_dictionary[char] for char in data[0]] for data in datas]
    y = [[label_dictionary[label] for label in data[1]] for data in datas]
    x = pad_sequences(x, TIME_STAMPS, padding='post', value=0)
    y = pad_sequences(y, TIME_STAMPS, padding='post', value=0)
    y = np.expand_dims(y, 2)
    return x, y