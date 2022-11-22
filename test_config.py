import torch
class config():
    model = 'Bert_CNN'
    version = None
    epoch = 40
    batch_size = 600
    batch_num = 20
    test_batch_size = 2000
    lr = 0.08
    lr_decay = 0.95
    lr_decay_steps = 10
    regular = 1
    if_ask = False
    max_seq_length = 384
    label_list = torch.LongTensor([1, 2, 3, 4, 5])