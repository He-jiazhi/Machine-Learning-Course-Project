import csv
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from modeling import *
from data_generate import *
from test_config import config

DATA_PATH = "./test2.csv"
SAVE_PATH = "./PB18000061_result2.txt"

# 调试用
# DATA_PATH = "./MediaPlayer.csv"
# SAVE_PATH = "./PB18000061_result1.txt"

class PB18000061():
    def predict(self, input_ids): 
        CNN = Model()
        CNN.add(conv2d('Convolution_1' ,24, [4, 768]))
        CNN.add(todense('todense'))
        CNN.add(batch_norm('Batch_Norm_1', tanh()))
        CNN.add(dense('Dense_1', 64, tanh()))
        CNN.add(batch_norm('Batch_Norm_2'))
        CNN.add(dense('Dense_2', 32, tanh()))
        CNN.add(dense('Dense_4', 1, sigmoid()))
        CNN.add(Linear('Linear'))

        CNN.add(dense('Dense_5', 5, tanh()))
        CNN.add(dense('Dense_6', 5, sigmoid()))
        CNN.build(config = config())

        CNN.load('Bert_CNN_4.2.9')
        pred = CNN.predict(input_ids).tolist()

        return pred


## for local validation
if __name__ == '__main__':
    data = Data()
    data.Read_Data_from_file(DATA_PATH)
    data.Data_Process(if_for_train=False)
    input_ids = data.input_ids

    pred = PB18000061().predict(input_ids)

    f = open(SAVE_PATH, 'w')
    for result in pred:
        f.write(str(result))
        f.write('\n')
    f.close()