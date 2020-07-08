# -*- coding: utf-8 -*-
from util import CONSTANTS
import pickle

'''生成word字典'''
def build_dict():
    vocabs = {'UNK'}
    for line in open("data/full.txt", 'r', encoding='utf-8'):
        line = line.rstrip().split('\t')
        if not line:
            continue
        char = line[0]
        if not char:
            continue
        if char is not '.':
            vocabs.add(char)
    word_dict = {wd: index for index, wd in enumerate(list(vocabs))}
    inverse_word_dic = {i: wd for i, wd in enumerate(list(vocabs))}
    print(word_dict.__len__())
    return word_dict, inverse_word_dic

# 保存字典
def save_dict():
    # 字典列表
    # labels = ['O','SUBJECT-B','SUBJECT-I','BODY-B','BODY-I','DECORATE-B','DECORATE-I','FREQUENCY-B','FREQUENCY-I','ITEM-B','ITEM-I']
    labels = ['O','SUBJECT-B','SUBJECT-I','BODY-B','BODY-I','DECORATE-B','DECORATE-I','FREQUENCY-B','FREQUENCY-I','ITEM-B','ITEM-I','DISEASE-B','DISEASE-I']
    word_dictionary, inverse_word_dictionary = build_dict()
    print("word_dictionary", word_dictionary)
    print("inverse_word_dictionary", inverse_word_dictionary)
    label_dictionary = {label: i for i, label in enumerate(labels)}
    print("label_dictionary",label_dictionary)
    output_dictionary = {i: labels for i, labels in enumerate(labels)}
    print("output", output_dictionary)

    dict_list = [inverse_word_dictionary, word_dictionary, label_dictionary, output_dictionary]

    #保存为pickle形式
    for dict_item, path in zip(dict_list, CONSTANTS[3:]):
        with open(path, 'wb') as f:
            pickle.dump(dict_item, f)

# build_dict()
# save_dict()
