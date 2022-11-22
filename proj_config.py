# 该文件用于存储路径

code_dir = './'

raw_data_dir = '%s%s' % (code_dir, 'train_data_all/')

save_data_dir = '%s%s' % (code_dir, 'data/')
save_data_X_train = '%s%s.csv' % (save_data_dir, 'X_train')
save_data_y_train = '%s%s.csv' % (save_data_dir, 'y_train')

save_model_dir = '%s%s' % (code_dir, 'model/')
word_embeddings_path = '%s%s' % (save_model_dir, 'bert_embeddings_word_embeddings.pth')
token_type_embeddings_path = '%s%s' % (save_model_dir, 'bert_embeddings_token_type_embeddings.npy')
position_embeddings_path = '%s%s' % (save_model_dir, 'bert_embeddings_position_embeddings.npy')

bert_dir = '%s%s' % (code_dir, 'uncased_L-12_H-768_A-12/')
vocab_file= '%s%s' % (bert_dir, 'vocab.txt')
bert_config_file = '%s%s' % (bert_dir, 'bert_config.json')
init_checkpoint = '%s%s' % (bert_dir, 'bert_model.ckpt')

output_dir = '%s%s' % (code_dir, 'output/')

features_path = '%s%s' % (output_dir, "features.pkl")
