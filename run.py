# -*- coding: utf-8 -*-
from model import *
from evaluate import *

#keras版本2.2.3
#生成训练集
x_train, y_train = modify_data(CONSTANTS[0])
#模型训练
model = train_model(x_train, y_train)
#生成测试集
x_test, y_test = modify_data(CONSTANTS[1])
#评估模型
evaluate_model(CONSTANTS[2], x_test, y_test)