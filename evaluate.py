# -*- coding: utf-8 -*-
from keras_contrib.metrics import crf_viterbi_accuracy
from util import *
from keras.models import load_model
import pickle
import numpy as np
from keras_contrib.layers.crf import CRF, crf_loss
from keras.preprocessing.sequence import pad_sequences

# 导入字典
with open(CONSTANTS[4], 'rb') as f:
    word_dictionary = pickle.load(f)
with open(CONSTANTS[3], 'rb') as f:
    inverse_word_dictionary = pickle.load(f)
with open(CONSTANTS[5], 'rb') as f:
    label_dictionary = pickle.load(f)
with open(CONSTANTS[6], 'rb') as f:
    output_dictionary = pickle.load(f)
vocab_size = len(word_dictionary.keys())
label_size = len(label_dictionary.keys())


input_shape = 300


def evaluate_model(model_save_path, x_test, y_test):
    model = load_model(model_save_path, custom_objects={"CRF": CRF, 'crf_loss': crf_loss,
                                                             'crf_viterbi_accuracy': crf_viterbi_accuracy})
    # 模型预测
    y_predict = model.predict(x_test)
    # 在测试集上的效果
    N = x_test.shape[0]  # 测试的条数
    avg_accuracy = 0  # 预测的平均准确率
    for start, end in zip(range(0, N, 1), range(1, N + 1, 1)):
        sentence = [inverse_word_dictionary[i] for i in x_test[start] if i != 0]
        y_predict = model.predict(x_test[start:end])
        input_sequences, output_sequences = [], []
        for i in range(0, len(y_predict[0])):
            output_sequences.append(np.argmax(y_predict[0][i]))
            input_sequences.append(np.argmax(y_test[start][i]))

        loss, accuracy = model.evaluate(x_test[start:end], y_test[start:end])
        print('Test Accuracy: loss = %0.6f accuracy = %0.2f%%' % (loss, accuracy * 100))
        avg_accuracy += accuracy
        output_sequences = ' '.join([output_dictionary[key] for key in output_sequences if key != 0]).split()
        input_sequences = ' '.join([output_dictionary[key] for key in input_sequences if key != 0]).split()
        print(sentence)
        print(output_sequences)
        print(input_sequences)
        # output_input_comparison = pd.DataFrame([sentence, output_sequences, input_sequences]).T
        # print(output_input_comparison.dropna())
        print('#' * 80)

    avg_accuracy /= N
    print("测试样本的平均预测准确率：%.2f%%." % (avg_accuracy * 100))

def text_evaluate():
    # 测试单个症状
    input_shape = 300
    sent = "血脑屏障通透性增大"
    new_x = [[word_dictionary[word] for word in sent]]
    x = pad_sequences(maxlen=input_shape, sequences=new_x, padding='post', value=0)
    print(len(sent))
    # 载入模型
    model_save_path = CONSTANTS[2]
    model = load_model(model_save_path, custom_objects={"CRF": CRF, 'crf_loss': crf_loss,
                                               'crf_viterbi_accuracy': crf_viterbi_accuracy})

    # 模型预测
    y_predict = model.predict(x)
    ner_tag = []
    for i in range(0, len(sent)):
        ner_tag.append(np.argmax(y_predict[0][i]))
    ner = [output_dictionary[i] for i in ner_tag]
    print(sent)
    print(ner)


if __name__ == '__main__':
    text_evaluate()





