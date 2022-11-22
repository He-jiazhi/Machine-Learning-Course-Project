import os
import pandas as pd
import tokenization
import torch
from proj_config import *

# 此文件用来产生训练数据
class Data():
    def __init__(self, max_seq_lenth = 384):
        self.max_seq_length = max_seq_lenth

    def Read_Data_from_dir(self, data_dir):
        # 从文件夹读入数据
        files= os.listdir(data_dir)

        raw_data = pd.DataFrame()
        for file_name in files:    #遍历文件
            file = "%s%s" % (raw_data_dir, file_name)
            data = pd.read_csv(file)    # 读取数据
            commodity_name = os.path.splitext(file_name)[0]    # 获取商品名称
            data['name'] = commodity_name
            raw_data = raw_data.append(data, ignore_index=True)
        self.RawData = raw_data

    def Read_Data_from_file(self, data_path):
        # 从文件读入数据
        raw_data = pd.read_csv(data_path)    # 读取数据
        self.RawData = raw_data

    def Data_Process(self, if_for_train):
        # 清洗数据
            # 重复项
            # 缺项
            # 剔除评论中非单词
        data = self.RawData

        # 训练时使用
        if if_for_train==True:
            data = data[['CommentsTitle', 'CommentsContent', 'PurchasemModel_Size', 'CommentsStars']]
            # 删除重复
            data = data.drop_duplicates()
            # 处理缺项
            # 商品属性缺项用*填补
            data['PurchasemModel_Size'] = data['PurchasemModel_Size'].fillna('*')
            # 评论标题、内容、星级缺项删除
            data = data.dropna(subset=['CommentsTitle','CommentsContent','CommentsStars'])
            data = data.reset_index(drop=True)
        # 检验时使用
        else:
            data = data[['CommentsTitle', 'CommentsContent', 'PurchasemModel_Size']]
            # 处理缺项
            data['CommentsTitle'] = data['CommentsTitle'].fillna('*')
            data['CommentsContent'] = data['CommentsContent'].fillna('*')
            data['PurchasemModel_Size'] = data['PurchasemModel_Size'].fillna('*')

        
        data_num = len(data)
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        
        input_ids_table = torch.zeros(data_num, self.max_seq_length, dtype=torch.long)
        if if_for_train:
            label_ids = torch.zeros(data_num, dtype=torch.long)
        else:
            label_ids = None

        # 遍历表格处理文本
        for item, comment in data.iterrows():
            if item % 1000 == 0:
                print("Processing example %d of %d" % (item, data_num))

            # 处理评论标题
            title = tokenizer.tokenize(comment['CommentsTitle'])
            
            # 处理评论内容
            content = tokenizer.tokenize(comment['CommentsContent'])

            # 处理商品属性
            model_size = tokenizer.tokenize(comment['PurchasemModel_Size'])

            # 处理评论星级
            if if_for_train:
                stars = comment['CommentsStars']
                if type(stars) is str:
                    stars = stars[0:3]    # 一些评论星级形如“1.0 颗星，最多 5 颗星”
                stars = int(float(stars))

            # 截断操作
            total_length = len(title) + len(content)
            if total_length > self.max_seq_length - 5:
                content = content[0:len(content) - total_length + self.max_seq_length - 5]
            
            tokens = ['[CLS]'] + title + ['[SEP]'] + content + ['[SEP]'] + model_size
            if len(tokens) > self.max_seq_length - 1:
                tokens = tokens[0:self.max_seq_length - 1]

            tokens = tokens + ['[SEP]']

            input_ids = tokenizer.convert_tokens_to_ids(tokens) #转换成ID

            # 补零操作
            while len(input_ids) < self.max_seq_length: #PAD的长度取决于设置的最大长度
                input_ids.append(0)

            assert len(input_ids) == self.max_seq_length

            input_ids_table[item] = torch.LongTensor(input_ids)

            if if_for_train:
                label_id = int(stars)
                label_ids[item] = label_id
        
        # 词id
        self.input_ids = torch.LongTensor(input_ids_table)
        if if_for_train:
            self.label_ids = torch.LongTensor(label_ids)
    
    def save(self):
        torch.save(self.input_ids, '%s%s.pth' % (output_dir, 'input_ids'))
        torch.save(self.label_ids, '%s%s.pth' % (output_dir, 'label_ids')) 

if __name__ == '__main__':
    data = Data()
    data.Read_Data_from_dir(raw_data_dir)
    data.Data_Process(if_for_train=True)
    data.save()
