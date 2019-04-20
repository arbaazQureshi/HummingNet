import numpy as  np

from keras.layers import Dense, Conv1D, Conv2D, Flatten, Input, Concatenate, Dropout, Subtract, CuDNNLSTM, CuDNNGRU
from keras.models import Model

from sklearn.metrics import f1_score

import os

os.environ["CUDA_VISIBLE_DEVICES"]="7"


all_training_queries = np.load('data/all_training_queries.npy')
all_training_songs = np.load('data/all_training_songs.npy')
all_training_labels = np.load('data/all_training_labels.npy')

all_testing_queries = np.load('data/all_testing_queries.npy')
all_testing_songs = np.load('data/all_testing_songs.npy')
all_testing_labels = np.load('data/all_testing_labels.npy')


embedder_lstm_1 = CuDNNGRU(300, input_shape = (20700,430,), return_sequences = True)
embedder_lstm_2 = CuDNNGRU(150)

q = Input(shape = (20700,430,))
s = Input(shape = (20700,430,))

q_embeddings = embedder_lstm_1(q)
q_embeddings = embedder_lstm_2(q_embeddings)

s_embeddings = embedder_lstm_1(s)
s_embeddings = embedder_lstm_2(s_embeddings)

final = Concatenate(axis = -1)([q_embeddings, s_embeddings])
#final = Subtract()([q_embeddings, s_embeddings])
label = Dense(90, activation = 'relu')(final)
label = Dropout(rate = 0.2)(label)
label = Dense(1, activation = 'sigmoid')(label)

model = Model(inputs = [q, s], outputs = label)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

model.fit([all_training_queries, all_training_songs], all_training_labels, validation_split = 0.20, epochs = 100, batch_size = 32)