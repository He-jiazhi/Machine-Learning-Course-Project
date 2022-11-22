from modeling import *
import os

# 该文件用于将线性回归模型转换为分类模型
class config():
    model = 'Bert_CNN'
    version = '5.1' # 额外跑的
    load_version = '5.0'
    epoch = 60
    batch_size = 1000
    batch_num = 10
    test_batch_size = 2000
    lr = 0.02
    lr_decay = 0.95
    lr_decay_steps = 10
    regular = 0.2
    if_ask = False
    max_seq_length = 384
    label_list = torch.LongTensor([1, 2, 3, 4, 5])

    # 1.1: lr=0.1, bias=0.5 (1/4) 较多4
    # 1.2: lr=0.1, bias=0.55(1/4) 仍然较多4， 但情况好转
    # 1.3: lr=0.08, bias=0.62 较多2和4
    # 1.5: 损失函数加入两侧较小的W型函数
    # 2.1: 强行向两端分流 
    # 分流较好
    # 2.5 尝试取二次项的损失函数    1-(p-0.6)(t-0.6) 失败了，
    # 2.6 回到原来的，加上到两端距离的项

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
    CNN.add(dense('Dense_6', 5, tanh()))
    CNN.add(Linear('Linear'))
    
    CNN.build(config = config())
    for layer in CNN.layers[:-2]:
        layer.load('Bert_CNN_' + config.load_version)

    CNN.layers[-2].kernel = torch.Tensor([[-0.4, -0.2, 0, 0.2, 0.4]])
    CNN.layers[-2].bias = torch.Tensor([0.24, 0.12, 0, -0.12, -0.24])
    CNN.layers[-1].kernel = torch.eye(5)
    CNN.layers[-1].bias = torch.zeros(5)

    CNN.train(train_X, train_y, test_X, test_y)