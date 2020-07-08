# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from keras_contrib.layers.crf import CRF
import os
from dataset import *
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#加载字典#
with open(CONSTANTS[4], 'rb') as f:
    word_dictionary = pickle.load(f)
with open(CONSTANTS[5], 'rb') as f:
    label_dictionary = pickle.load(f)
with open(CONSTANTS[6], 'rb') as f:
    output_dictionary = pickle.load(f)

#参数设置
EMBEDDING_DIM = 300
EPOCHS = 5
BATCH_SIZE = 32
NUM_CLASSES = len(label_dictionary)
VOCAB_SIZE = len(word_dictionary)
TIME_STAMPS = 300

callbacks_list = [
  keras.callbacks.EarlyStopping(
  monitor='val_crf_viterbi_accuracy',
  patience=1,
  ),

  keras.callbacks.ModelCheckpoint(
  filepath=CONSTANTS[2],
  monitor='val_loss',
  save_best_only=True,
  )
]



'''加载预训练词向量'''
def load_pretrained_embedding():
    embeddings_dict = {}
    with open('model/token_vec_300.bin', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split(' ')
            if len(values) < 300:
                continue
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = coefs
    print('Found %s word vectors.' % len(embeddings_dict))
    return embeddings_dict

'''加载词向量矩阵'''
def build_embedding_matrix():
    embedding_dict = load_pretrained_embedding()
    embedding_matrix = np.zeros((VOCAB_SIZE + 1, EMBEDDING_DIM))
    for word, i in word_dictionary.items():
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

'''使用预训练向量进行模型训练'''
def tokenvec_bilstm2_crf_model():
    model = Sequential()
    embedding_layer = Embedding(VOCAB_SIZE + 1,
                                    EMBEDDING_DIM,
                                    weights=[build_embedding_matrix()],
                                    input_length=TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=True)
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(NUM_CLASSES)))
    crf_layer = CRF(NUM_CLASSES, sparse_target=True)
    model.add(crf_layer)
    model.compile('adam', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
    model.summary()
    return model

'''训练模型'''
def train_model(x_train, y_train):
    model = tokenvec_bilstm2_crf_model()
    print(x_train[:].shape)
    x_dev, y_dev = modify_data("data/dev_data.txt")
    model.fit(x_train[:], y_train[:],validation_data=(x_dev, y_dev), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks_list)
    model.save(CONSTANTS[2])
    return model

if __name__ == '__main__':
    x_train, y_train = modify_data(CONSTANTS[0])
    train_model(x_train, y_train)