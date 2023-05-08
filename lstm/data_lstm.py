from keras.preprocessing import text, sequence
import pandas as pd
import numpy as np
import torch
import tqdm
from torch.utils import data
GLOVE_EMBEDDING_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'
MAX_LEN = 220
BATCH_SIZE = 512

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))

def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words

train = pd.read_csv('data/df_treated_comment.csv')
x_train = train['treated_comment']
y_train = np.where(train['target'] >= 0.5, 1, 0)
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(x_train))

x_train = tokenizer.texts_to_sequences(x_train)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
max_features = None or len(tokenizer.word_index) + 1
glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
print('n unknown words (glove): ', len(unknown_words_glove))
x_train_torch = torch.tensor(x_train, dtype=torch.long).cuda()
y_train_torch = torch.tensor(y_train, dtype=torch.float32).cuda() 
train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
 