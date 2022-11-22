from torch.serialization import load
from modeling import *
import os

class config():
    # 模型超参数
    model = 'Bert_CNN'
    version = '4.2.9'
    load_version = '4.2.8'
    epoch = 20
    batch_size = 600
    batch_num = 10
    test_batch_size = 2000
    lr = 0.15
    lr_decay = 0.95
    lr_decay_steps = 10
    regular = 0.1
    if_ask = False
    if_train = True
    max_seq_length = 384
    label_list = torch.LongTensor([1, 2, 3, 4, 5])

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 读取数据
    input_ids = torch.load('%s%s.pth' % (output_dir, 'input_ids'))
    label_ids = torch.load('%s%s.pth' % (output_dir, 'label_ids'))

    init_sample_num = len(label_ids)

    # 数据随机打乱
    index_shuffle = torch.randperm(init_sample_num)
    input_ids = input_ids[index_shuffle]
    label_ids = label_ids[index_shuffle]

    # 为了处理数据不平衡问题，对数据进行重新采样
    index = []
    for i in [1, 2, 3, 4, 5]:
        index.append(np.where(label_ids == i))
    sample_size_per_class = min(len(index[i][0]) for i in range(5))

    new_index = []
    for i in range(5):
        index[i] = index[i][0][0:sample_size_per_class]
        new_index = np.concatenate((new_index, index[i]),axis=0)
    np.random.shuffle(new_index)

    sample_num = len(new_index)

    # 划分训练集和回测集
    train_num = int(sample_num * 0.8)
    input_ids = input_ids[new_index]
    label = label_ids[new_index]

    train_X = input_ids[0:train_num]
    train_y = label[0:train_num]

    test_X = input_ids[train_num:]
    test_y = label[train_num:]

    label_list = torch.LongTensor([1, 2, 3, 4, 5])

    # 搭建模型
    CNN = Model()
    CNN.add(conv2d('Convolution_1' ,24, [4, 768]))
    CNN.add(todense('todense'))
    CNN.add(batch_norm('Batch_Norm_1', tanh()))
    CNN.add(dense('Dense_1', 64, tanh()))
    CNN.add(batch_norm('Batch_Norm_2'))
    CNN.add(dense('Dense_2', 32, tanh()))
    CNN.add(dense('Dense_4', 1, sigmoid()))
    CNN.add(Linear('Linear'))

    # 以下为分类模型，为后期添加
    CNN.add(dense('Dense_5', 5, tanh()))
    CNN.add(dense('Dense_6', 5, tanh()))

    CNN.build(config = config())
    if config.load_version is not None:
        CNN.load('Bert_CNN_' + config.load_version)
    CNN.train(train_X, train_y, test_X, test_y)
    