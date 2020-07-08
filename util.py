
# 存放字典、模型等路径
BASE_DIR = "dict"

TRAIN_DATA_PATH = "data/train3.txt"
WORD_DICTIONARY_PATH = '%s/word_dictionary.pk' % BASE_DIR
INVERSE_WORD_DICTIONARY_PATH = '%s/inverse_word_dictionary.pk' % BASE_DIR
LABEL_DICTIONARY_PATH = '%s/label_dictionary.pk' % BASE_DIR
OUTPUT_DICTIONARY_PATH = '%s/output_dictionary.pk' % BASE_DIR
SAVE_MODEL_PATH = "model/mymodel.h5"
TEST_DATA_PATH = "data/test_data.txt"

CONSTANTS = [
             TRAIN_DATA_PATH,
             TEST_DATA_PATH,
             SAVE_MODEL_PATH,
             INVERSE_WORD_DICTIONARY_PATH,
             WORD_DICTIONARY_PATH,
             LABEL_DICTIONARY_PATH,
             OUTPUT_DICTIONARY_PATH,
             ]