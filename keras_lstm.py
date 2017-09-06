import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

d_train = pd.read_csv("train.csv").dropna()
d_test = pd.read_csv("test.csv").dropna()

X_Train = d_train.drop('class', axis=1)
Y_Train = d_train['class']

X_Test = d_test.drop('1', axis=1)
Y_Test = d_test['1']

lb = LabelBinarizer()
Y_Train = lb.fit_transform(Y_Train)
Y_Test = lb.transform(Y_Test)

tkn = Tokenizer(lower=False)
tkn.fit_on_texts(np.concatenate((X_Train, X_Test)).ravel())

X_Train_tokenized = pad_sequences(tkn.texts_to_sequences(X_Train.values.ravel()))
X_Test_tokenized = pad_sequences(tkn.texts_to_sequences(X_Test.values.ravel()))

embeddings = 128
lstm_units = 32

ES = EarlyStopping(patience=1)
RL = ReduceLROnPlateau(patience=1)

model = Sequential()
model.add(Embedding(len(tkn.word_index), embeddings, input_length=X_Train_tokenized.shape[1]))
model.add(Bidirectional(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.3)))
model.add(BatchNormalization())
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

model.fit(X_Train_tokenized, Y_Train, callbacks=[RL, ES], epochs=100, validation_split=0.15)

Y_Pred = model.predict_classes(X_Test_tokenized)

print("\n")
print("LSTM Accuracy:")
print(accuracy_score(Y_Test.argmax(-1), Y_Pred))

probabilities = pd.DataFrame(model.predict_proba(X_Test_tokenized), columns=lb.classes_)
probabilities.to_csv("lstm_probabilities.csv", index=False)