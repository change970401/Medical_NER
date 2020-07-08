# -*- coding: utf-8 -*-
import os
import json

class TransferData:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])

        self.cate_dict ={
                         'O':0,
                         'SUBJECT-B': 1,
                         'SUBJECT-I': 2,
                         'BODY-B': 3,
                         'BODY-I': 4,
                         'DECORATE-B': 5,
                         'DECORATE-I': 6,
                         'FREQUENCY-B': 7,
                         'FREQUENCY-I': 8,
                         'ITEM-B': 9,
                         'ITEM-I': 10
                        }
        self.origin_path = os.path.join(cur, 'data/train1.txt')
        self.train_filepath = os.path.join(cur, 'data/train2.txt')
        return


    def transfer(self):
        f = open(self.train_filepath, 'w+', encoding='utf-8')
        fileread = open(self.origin_path, 'r', encoding='utf-8')
        contents = json.load(fileread)
        for content in contents:
            for symptom in list(content['symptom'].values()):
                print(symptom['self'])
                if(symptom['self']['pos'].__len__() == 2):
                    start = int(symptom['self']['pos'][0])
                    end = int(symptom['self']['pos'][1])
                res_dict = {}
                for key in symptom.keys():
                    if(key == 'self' or key == 'disease' or key == 'has_problem'):
                        continue
                    '''pos长度大于2'''
                    if (symptom[key]['pos'].__len__() >= 2):
                        all_pos = symptom[key]['pos']
                        pos_list = list(zip(*[iter(all_pos)] * 2))
                        for pos in pos_list:
                            feature_start = int(pos[0])
                            feature_end = int(pos[1])
                            if(not(feature_start>=start and feature_end<=end)):
                                continue
                            '''计算相对下标index'''
                            rel_start = feature_start - start
                            rel_end = feature_end - start
                            label_id = key.upper()
                            for i in range(rel_start, rel_end+1):
                                if i == rel_start:
                                    '''label起始位置用-B表示'''
                                    label = label_id + '-B'
                                else:
                                    label = label_id + '-I'
                                res_dict[i] = label
                for indx, char in enumerate(symptom['self']['val']):
                    char_label = res_dict.get(indx, 'O')
                    print(char, char_label)
                    f.write(char + '\t' + char_label + '\n')
                f.write('.' + '\n')

        f.close()
        return




if __name__ == '__main__':
    handler = TransferData()
    train_datas = handler.transfer()