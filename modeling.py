from proj_config import *
import torch
import torch.nn.functional as F
import numpy as np
import time
from math import sqrt
from sklearn.metrics import f1_score

class activation_function():    # 定义激活函数类
    def __init__(self):  pass
    
    def value(self):    raise NotImplemented

    def grad(self):     raise NotImplemented

class linear(activation_function):
    def value(self, x):     return x
    def grad(self, x):      return torch.ones_like(x)

class sigmoid(activation_function):
    def value(self, x):     return 1 / (1 + torch.exp(-x))
    def grad(self, x):      return torch.mul(self.value(x), ( 1 - self.value(x)))

class relu(activation_function):
    def value(self, x):     return torch.where(x > 0, x, torch.zeros_like(x))
    def grad(sel, x):       return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

class tanh(activation_function):
    def value(self, x):     return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    def grad(self, x):      return 1 - torch.pow(self.value(x), 2)


class Layer():     # 定义层类
    def __init__():
        raise NotImplementedError
    
    def initialize():     # 参数初始化
        raise NotImplementedError

    def forward():      # 向前传播
        raise NotImplementedError

    def backward():     # 向后传播
        raise NotImplementedError
    
    def call():     # 计算值
        raise NotImplementedError
    
    def save(self):     # 保存
        pass

    def load(self, model_name):     # 加载
        pass

class embedding(Layer):     # 将词id转成词向量
    def __init__(self):
        self.layer_name = 'embedding'
        self.embedding_table = torch.load(word_embeddings_path)     # 加载预训练的词向量
        self.embedding_size = 768

    def initialize(self, input_size, model_name):
        self.input_size = input_size
        self.name = '%s_%s' % (model_name, self.layer_name)
        self.output_size = torch.Size([input_size[0], 1, input_size[1], self.embedding_size])

    def forward(self, input_ids):
        self.inputs = input_ids
        
        flat_input_ids = torch.reshape(input_ids, [-1])
        output = self.embedding_table.index_select(0, flat_input_ids)  # 计算一个batch里所有词的向量

        input_shape = input_ids.shape       # 将输出的向量转换为4维，方便卷积层操作
        self.outputs = torch.reshape(output, [input_shape[0], 1, input_shape[1], self.embedding_size]) # [B,1,384,768]
        return self.outputs

    def backward(self, _):
        return None

    def call(self, input_ids):
        flat_input_ids = torch.reshape(input_ids, [-1])
        output = self.embedding_table.index_select(0, flat_input_ids)

        input_shape = input_ids.shape
        return torch.reshape(output, [input_shape[0], 1, input_shape[1], self.embedding_size])

class conv2d(Layer):    # 二维卷积层
    def __init__(self,
                name,
                filters_num, # filter的数量
                filter_size, # [FH, FW]
                activation=linear(),
                if_back=False,  # 当前面没有需要更新参数的层的时候，不再计算向前传播的导数
                ):
        self.layer_name = name
        self.filters_num = filters_num
        self.FH = filter_size[0]
        self.FW = filter_size[1]
        self.activation = activation
        self.if_back=if_back
    
    def initialize(self, input_size, model_name):
        self.input_size = input_size    # [B, C, H, W]
        self.name = '%s_%s' % (model_name, self.layer_name)

        self.H, self.W = (input_size[2], input_size[3]) 
        self.output_size = torch.Size([input_size[0], self.filters_num, self.H - self.FH + 1, self.W - self.FW + 1])
        self.filters = torch.randn(self.filters_num, self.input_size[1], self.FH, self.FW) / 100
        self.bias = torch.zeros(self.filters_num, self.H - self.FH + 1, self.W - self.FW + 1)
    
    def forward(self, inputs):
        # input: [B, C, H, W]
        # weight: [FN, C, FH, FW]
        # bias: [FN, OH, OW]
        # output: [B, FN, OH, OW]
        self.inputs = inputs
        self.hidden = F.conv2d(inputs, self.filters) + self.bias    # 做卷积并加偏置
        self.outputs = self.activation.value(self.hidden)       # 激活函数
        return self.outputs

    def backward(self, delta, lr):
        # delta: [B, FN, OH, OW]
        delta = torch.mul(self.activation.grad(self.hidden), delta)

        # 更新参数
        bias_grad = torch.sum(delta, 0)
        self.bias -= lr * bias_grad
        
        inputs = torch.sum(self.inputs, 0, keepdim=True)
        delta = torch.sum(torch.transpose(delta, 0, 1), 1, keepdim=True)
        filter_grad = torch.transpose(F.conv2d(inputs, delta), 0, 1)
        self.filters -= lr * filter_grad

        # 向前传播
        # 这一过程需要将导数补零并和旋转后的权重做卷积， 推导过程见report
        # 由于计算非常缓慢，当前面没有需要更新参数的层时，不做这一步计算
        next_delta = None
        if self.if_back:
            # 导数补零
            pdH = self.FH - 1
            pdW = self.FW - 1
            pad_delta = F.pad(delta, pad=(pdW,pdW,pdH,pdH), mode='constant')
            pad_delta = torch.reshape(pad_delta, [self.input_size[0], self.filters_num, pad_delta.shape[2], pad_delta.shape[3]])
            # 旋转权重函数
            weight_inverse = torch.flip(self.filters, [2,3])
            weight_inverse = torch.transpose(weight_inverse, 0, 1)

            next_delta = F.conv2d(pad_delta, weight_inverse)

        return next_delta
    
    def call(self, inputs):
        outputs = self.activation.value(F.conv2d(inputs, self.filters) + self.bias)
        return outputs
    
    def save(self):
        torch.save(self.filters, ('%s%s_%s.pth' % (save_model_dir, self.name, 'filters')))
        torch.save(self.bias, ('%s%s_%s.pth' % (save_model_dir, self.name, 'bias')))

    def load(self, model_name):
        self.filters = torch.load('%s%s_%s_%s.pth' % (save_model_dir, model_name, self.layer_name, 'filters'))
        self.bias = torch.load('%s%s_%s_%s.pth' % (save_model_dir, model_name, self.layer_name, 'bias'))

class todense(Layer):   # 将卷积的结果转为2维，传入后面的全连接层
    def __init__(self, name):
        self.layer_name = name

    def initialize(self, input_size, model_name):
        self.input_size = input_size
        self.name = '%s_%s' % (model_name, self.layer_name)
        self.output_size = torch.Size([input_size[0], input_size[1] * input_size[2] * input_size[3]])

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = torch.reshape(self.inputs, self.output_size)
        return self.outputs

    def backward(self, delta, _):
        next_delta = torch.reshape(delta, self.input_size)
        return next_delta
    
    def call(self, inputs):
        self.inputs = inputs
        input_size = self.inputs.shape
        self.outputs = torch.reshape(inputs, [input_size[0], input_size[1] * input_size[2] * input_size[3]])
        return self.outputs

class pooling_1d(Layer):    # 一维池化层
    def __init__(self, name, window_size = 2):
        self.layer_name = name
        self.window_size = window_size

    def initialize(self, input_size, model_name):
        self.input_size = input_size
        self.name = '%s_%s' % (model_name, self.layer_name)
        self.inL = input_size[1]
        self.outL = int(self.inL / self.window_size)
        self.output_size = torch.Size([self.input_size[0], self.outL])

    def forward(self, inputs):
        self.inputs = inputs
        unpooled = torch.unsqueeze(inputs, 1)
        outputs, self.indices = F.max_pool1d_with_indices(unpooled, self.window_size)
        self.outputs = torch.reshape(outputs, self.output_size)
        # 以下是我自己实现的池化操作，但计算很慢，因此改用torch.nn.funtional中的池化函数
        '''
        self.weight = torch.zeros_like(inputs)
        self.outputs = torch.zeros([self.batch_size, self.outL])
        L = self.outL * self.window_size
        pad_inputs = F.pad(inputs, [0, L - self.inL])
        ones = torch.eye(inputs.shape[0], self.window_size)
        for i in range(self.outL):
            pooled, index = torch.max(pad_inputs[:,i * self.window_size: (i + 1) * self.window_size], 1)
            one_hot = torch.index_select(ones, 0, index=index)
            self.weight[:, i * self.window_size: (i + 1) * self.window_size] = one_hot
            self.outputs[:,i] = pooled
        self.outputs = torch.reshape(self.outputs, self.output_size)
        '''
        return self.outputs

    def backward(self, delta, _):
        delta = torch.unsqueeze(delta, 1)
        unpooled = torch.squeeze(F.max_unpool1d(delta, self.indices, self.window_size))
        next_delta = F.pad(unpooled, [0, self.inL - self.outL * self.window_size])
        # 以下是我自己实现的池化计算，因为计算较慢而不采用
        '''
        def unpooling1d(matrix, w):
            out = torch.zeros(matrix.shape[0], matrix.shape[1] * w)
            column = list(range(matrix.shape[1]))
            for j in range(w):
                out[:, w * column + j] = matrix[:, column]
            return out
        unpooled = unpooling1d(delta, self.window_size)
        next_delta = torch.mul(self.weight, unpooled)[:, 0:self.inL] 
        '''
        return next_delta
    
    def call(self, inputs):
        inputs =  torch.unsqueeze(inputs, 1)
        outputs = torch.squeeze(F.max_pool1d(inputs, self.window_size, return_indices = False))
        outputs = torch.squeeze(outputs)
        return outputs

class batch_norm(Layer):    # 标准化层
    def __init__(self,
                name,
                activation=linear()
                ):
        self.layer_name = name
        self.activation = activation
    
    def initialize(self, input_size,  model_name):
        self.input_size = input_size
        self.name = '%s_%s' % (model_name, self.layer_name)
        self.output_size = self.input_size

        self.gamma = torch.ones(1)
        self.beta = torch.zeros(1)
        self.epsilon = 0.000001     # 防止分母出现零
    
    def forward(self, inputs):
        # input: [B, input_size]
        # output: [B, units]
        self.inputs = inputs
        self.mean = torch.mean(self.inputs, 0)
        self.sigma = torch.sqrt(torch.mean(torch.pow(self.inputs - self.mean, 2), 0) + self.epsilon)
        self.xhat = (self.inputs - self.mean) / self.sigma     # 保留xhat方便计算向前传播
        self.hidden = self.gamma * self.xhat + self.beta
        self.outputs = self.activation.value(self.xhat)
        return self.outputs

    def backward(self, delta, lr):
        # delta: [units]
        delta = torch.mul(self.activation.grad(self.hidden), delta)    # 激活层导数
        gamma_grad = torch.sum(torch.sum(torch.mul(delta, self.xhat),0))
        beta_grad = torch.sum(torch.sum(delta, 0))
        xhat_grad = delta * self.gamma

        # 根据公式推导的导数如下，但实际上可以化简
        '''
        sigma2_grad = - 0.5 * torch.sum(torch.mean(torch.mul(xhat_grad, 
                        (self.inputs - self.mean)),0)) * torch.pow(self.sigma, -3)

        mean_grad = - torch.sum(torch.mean(xhat_grad,0)) / (self.sigma + self.epsilon) - 2 / self.input_size[1] * torch.sum(
                        torch.mean(self.inputs - self.mean, 0)) * sigma2_grad

        next_delta = xhat_grad / (self.sigma + self.epsilon) + 2 / self.input_size[1] * sigma2_grad * (
                self.inputs - self.mean) + mean_grad / self.input_size[1]
        '''

        next_delta = xhat_grad / self.sigma

        self.gamma -= lr * gamma_grad
        self.beta -= lr * beta_grad

        return next_delta
    
    def call(self, inputs):
        mean = torch.mean(inputs, 0)
        sigma = torch.sqrt(torch.mean(torch.pow(inputs - mean, 2), 0) + self.epsilon)
        return self.activation.value(self.gamma * (inputs - mean) / sigma + self.beta)

    def save(self):
        torch.save(self.gamma, ('%s%s_%s.pth' % (save_model_dir, self.name, 'gamma')))
        torch.save(self.beta, ('%s%s_%s.pth' % (save_model_dir, self.name, 'beta')))
    
    def load(self, model_name):
        self.gamma = torch.load('%s%s_%s_%s.pth' % (save_model_dir, model_name, self.layer_name, 'gamma'))
        self.beta = torch.load('%s%s_%s_%s.pth' % (save_model_dir, model_name, self.layer_name, 'beta'))

class dense(Layer):     # 全连接层
    def __init__(self,
                name,
                units,
                activation=linear()
                ):
        self.layer_name = name
        self.units = units
        self.activation = activation
    
    def initialize(self, input_size, model_name):
        self.input_size = input_size
        self.name = '%s_%s' % (model_name, self.layer_name)
        self.output_size = torch.Size([self.input_size[0], self.units])
        self.kernel = torch.randn(self.input_size[1], self.units) / sqrt(self.input_size[0])    # 随机初始化，使输出后的方差保持1
        self.bias = torch.randn(self.units) / sqrt(self.input_size[0])
    
    def forward(self, inputs):
        # input: [B, input_size]
        # output: [B, units]
        self.inputs = inputs
        self.hidden = torch.matmul(inputs, self.kernel) + self.bias
        self.outputs = self.activation.value(self.hidden)
        return self.outputs

    def backward(self, delta, lr):
        # delta: [units]
        delta = torch.mul(self.activation.grad(self.hidden), delta)    # 激活层导数
        kernel_grad = torch.matmul(torch.transpose(self.inputs, -1, 0), delta)  # kernel的导数
        bias_grad = torch.sum(delta, 0)     # bias的导数

        self.kernel -= lr * kernel_grad
        self.bias -= lr * bias_grad

        next_delta = torch.matmul(delta, torch.transpose(self.kernel, -1, 0))   # 向前传递的导数
        next_delta = torch.reshape(next_delta, self.input_size)
        return next_delta
    
    def call(self, inputs):
        return self.activation.value(torch.matmul(inputs, self.kernel) + self.bias)

    def save(self):
        torch.save(self.kernel, ('%s%s_%s.pth' % (save_model_dir, self.name, 'kernel')))
        torch.save(self.bias, ('%s%s_%s.pth' % (save_model_dir, self.name, 'bias')))
    
    def load(self, model_name):
        self.kernel = torch.load('%s%s_%s_%s.pth' % (save_model_dir, model_name, self.layer_name, 'kernel'))
        self.bias = torch.load('%s%s_%s_%s.pth' % (save_model_dir, model_name, self.layer_name, 'bias'))

class Linear(Layer):    # 线性层
    def __init__(self,
                name,
                activation=linear()
                ):
        self.layer_name = name
        self.activation = activation
    
    def initialize(self, input_size, model_name):
        self.input_size = input_size
        self.name = '%s_%s' % (model_name, self.layer_name)
        self.output_size = self.input_size
        self.scaler = 1
        self.bias = 0
    
    def forward(self, inputs):
        self.inputs = inputs
        self.hidden = inputs * self.scaler + self.bias
        self.outputs = self.activation.value(self.hidden)
        return self.outputs

    def backward(self, delta, lr):
        delta = torch.mul(self.activation.grad(self.hidden), delta)    # 激活层导数
        scaler_grad = sum(torch.mul(delta, self.hidden))  # scaler的导数
        bias_grad = torch.sum(delta)     # bias的导数

        self.scaler -= lr * scaler_grad
        self.bias -= lr * bias_grad

        next_delta = delta * self.scaler   # 向前传递的导数
        return next_delta
    
    def call(self, inputs):
        return self.activation.value(inputs * self.scaler + self.bias)

    def save(self):
        torch.save(self.scaler, ('%s%s_%s.pth' % (save_model_dir, self.name, 'scaler')))
        torch.save(self.bias, ('%s%s_%s.pth' % (save_model_dir, self.name, 'bias')))
    
    def load(self, model_name):
        self.kernel = torch.load('%s%s_%s_%s.pth' % (save_model_dir, model_name, self.layer_name, 'scaler'))
        self.bias = torch.load('%s%s_%s_%s.pth' % (save_model_dir, model_name, self.layer_name, 'bias'))

class Model():     # 定义模型类
    def __init__(self):
        # 初始化时加入embedding层
        self.layers = [embedding()]

    def add(self, Layer):
        # 添加层
        self.layers.append(Layer)
    
    def build(self, config):
        # 构建模型并完成参数初始化
        # config为模型的一些超参数，在预测时指定，其含义在训练文件中说明
        self.model_name = '%s_%s' % (config.model, config.version)
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.lr = config.lr
        self.max_seq_length = config.max_seq_length
        self.if_ask = config.if_ask
        self.label_list = config.label_list
        self.batch_num = config.batch_num
        self.lr_decay = config.lr_decay
        self.lr_decay_steps = config.lr_decay_steps
        self.regular = config.regular
        self.label_num = self.label_list.shape[0]

        print('********************Building the Model********************')

        # 遍历所有层，构建模型并进行初始化
        input_size = torch.Size([self.batch_size, self.max_seq_length])     # batch输入形状 [B,M]
        num_layer = 1
        for layer in self.layers:
            # 层初始化
            layer.initialize(input_size, self.model_name)
            input_size = layer.output_size
            # 打印出入张量维度
            print('Layer %s : %s    input_size: %s    output_size: %s' % 
                    (num_layer, layer.layer_name, layer.input_size, layer.output_size))
            num_layer += 1
        
    def train(self, train_X, train_y, test_X, test_y):
        # 执行训练任务
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

        self.train_num = train_X.shape[0]
        self.test_num = test_X.shape[0]

        # 根据y值制作one_hot，用于计算softmax求导
        ones = torch.eye(self.label_num)
        self.one_hot = torch.index_select(ones, 0, index=train_y - 1)

        print('********************Begin traning********************')

        # 遍历epoch
        for epoch in range(self.epoch):

            print (time.strftime("---%H:%M:%S---    Epoch ", time.localtime()), epoch + 1)    # 打印Epoch

            if epoch + 1 % self.lr_decay_steps == 0:    # 学习率衰减
                self.lr *= self.lr_decay

            # 训练样本打乱
            index_shuffle = torch.randperm(self.train_num)

            for iteration in range(self.batch_num):     # 遍历循环
                
                # 取出batch训练样本
                batch_train_index = index_shuffle[iteration * self.batch_size:iteration * self.batch_size + self.batch_size]
                batch_test_index = np.random.choice(range(self.test_num), self.test_batch_size)
                
                # 正向传播
                outputs = self.train_X[batch_train_index]
                for layer in self.layers:
                    outputs = layer.forward(outputs)
                self.outputs = outputs

                # 计算Loss值和Loss对输出量的导数
                loss = self.train_loss(batch_train_index)
                delta = self.loss_grad

                # 计算在测试集上的acc
                test_acc = self.acc(self.test_X[batch_test_index], self.test_y[batch_test_index])
                
                # 打印训练信息
                print('Epoch: %s/%s,    Iteration: %s/%s,    Loss: %s,     train_F_score: %s,    test_F_score: %s' %
                        (epoch + 1, self.epoch, iteration + 1, self.batch_num, loss, self.train_acc, test_acc))
                
                # 打印各个样本预测数量
                # 这是由于模型经常出现对各类预测数量不平衡的现象
                pred_num = []
                true_num = []
                for i in [1, 2, 3, 4, 5]:
                    pred_num.append(self.pred.tolist().count(i))
                    true_num.append(self.test_y[batch_test_index].tolist().count(i))
                print('pre_num:', pred_num, 'true_num:', true_num)

                # 反向传播更新参数
                for layer in reversed(self.layers):
                    delta = layer.backward(delta, self.lr)

            # 每个epoch保存一次模型
            self.save()
    
    def train_loss(self, batch_index):
        # 训练损失函数
        '''
        # 前期处理回归问题
        pred = torch.squeeze(self.outputs)  # 预测结果 [B]
        true = self.train_y[batch_index]    # 真实值   [B]
        targ = (true - 1) / 4      # 目标回归值

        # 下面的损失函数在线性回归的第一阶段用
        each_loss = torch.abs(pred - targ) * 10
        self.loss = torch.mean(each_loss).item()
        self.loss_grad = torch.unsqueeze( 10 * torch.sign(pred - targ) / self.batch_size, 1)
        
        # 下面的损失函数在线性回归的第二阶段用，加上非凸的激励项，是为了鼓励模型预测极端值，但结果会较多预测2星和4星
        courage = torch.abs(torch.abs(pred-0.5) - 0.2)
        each_loss = (torch.abs(pred - targ) - self.regular * courage ) * 10
        self.loss = torch.mean(each_loss).item()
        self.loss_grad = torch.sgn(pred - targ) + torch.where()

        # 下面的损失函数在线性回归的第三阶段用，将结果向两端分流
        score = (torch.abs(pred - targ) + torch.min(torch.abs(pred + 1), torch.abs(pred - 2)) * self.regular ) * 10
        each_loss = torch.where( score > 0.0, score, torch.zeros_like(score))
        self.loss = torch.mean(each_loss).item()
        score_grad = torch.sgn(pred - targ) + torch.where(torch.abs(pred + 0.8) < torch.abs(pred - 2), torch.sign(pred + 1), torch.sign(pred - 2)) * self.regular
        self.loss_grad = torch.where(score > 0, score_grad, torch.zeros_like(score)) * 10 / self.batch_size
        self.loss_grad = torch.unsqueeze( self.loss_grad, 1)

        # 线性回归问题， 将结果转换为真实预测值
        pred = (pred * 4 + 1).tolist()
        pred = np.round(pred)
        for predi in pred:
            if predi > 5: predi = 5
            if predi < 1: predi = 1
        pred = torch.Tensor(pred)
        '''
        
        # 下面的损失函数在转变为分类问题时使用
        softmax = torch.nn.functional.softmax(self.outputs, 1)   # softmax [B, 5]
        log_softmax = torch.log(softmax)    # log p
        targets = self.train_y[batch_index] - 1     # 提取目标值并转化成索引

        each_loss = - torch.gather(log_softmax, 1, index=torch.unsqueeze(targets, 1))     # 交叉熵损失函数

        self.loss = torch.mean(each_loss).item()
        self.loss_grad = (softmax - self.one_hot[batch_index] ) / self.batch_size    # 导数传递

        # 计算模型f-score
        pred = torch.unsqueeze(self.predict(self.train_X[batch_index]), 1)    # 预测值
        true = torch.unsqueeze(self.train_y[batch_index], 1)    # 真实值

        self.train_acc = f1_score(y_true=true, y_pred=pred, labels=self.label_list, average="macro").item()
        return self.loss

    def predict(self, X):
        #计算给定样本X的预测值
        outputs = X
        for layer in self.layers:    # 样本通过构建的模型进行输出
            outputs = layer.call(outputs)
        
        '''
        # 线性回归模型的预测
        pred = torch.squeeze(outputs).tolist()

        def topred(num):    # 将预测结果转化成真实值
            num = round(num * 4 + 1)
            if num > 5: num = 5
            if num < 1: num = 1
            return num

        pred = torch.tensor(list(map(topred, pred)))    # 将预测结果转化成真实值
        '''

        # 分类问题的预测
        _, pre_y = torch.max(outputs, 1)    # 选取最大概率的结果
        pred = pre_y + 1    # 计算预测的星数

        return pred

    def acc(self, X, y):
        # 用于评估验证精确度
        self.pred = self.predict(X)    # 计算预测值
        return f1_score(y_true=y, y_pred=self.pred, labels=self.label_list, average="macro")    # 计算f1_score

    def save(self):
        # 保存模型
        for layer in self.layers:
            layer.save()
        print('The training results have been saved!')
    
    def load(self, model_name):
        # 加载训练好的模型
        for layer in self.layers:
            layer.load(model_name)
