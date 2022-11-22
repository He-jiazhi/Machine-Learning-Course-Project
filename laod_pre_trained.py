import tensorflow as tf
import torch
from tensorflow.python.training.py_checkpoint_reader import NewCheckpointReader
from proj_config import *

# 该文件读取Bert模型训练好的词向量，并保存为可用pytorch处理的pth格式

init_checkpoint = './uncased_L-12_H-768_A-12/bert_model.ckpt'

NewCheck =tf.compat.v1.train.NewCheckpointReader(init_checkpoint)
init_vars = tf.train.list_variables(init_checkpoint)

for x in init_vars:
    name = x[0]
    var = torch.Tensor(NewCheck.get_tensor(name))
    name = name.replace('/', '_')
    save_path = '%s%s.pth' % (save_model_dir, name)
    torch.save(var, save_path)
