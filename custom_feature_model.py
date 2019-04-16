# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle
from collections import Counter

from imblearn.over_sampling import SMOTE, RandomOverSampler

# Importing the dataset
dataset = pd.read_csv("data/names_combined.csv")

dataset.dropna(inplace=True)
dataset.replace('b','black',inplace=True)

# Cleaning the texts
import re
import nltk
corpus = []

for i in tqdm(range(len(dataset['name']))):
    try:
        review = re.sub('[^a-zA-Z]', ' ', dataset['name'].values[i])
        review = review.lower()
        corpus.append(review)
    except:
        print(dataset['name'][i])


import string
chars = string.ascii_lowercase + ' ' + '#'
char_dict = {}

for i in range(len(chars)):
	char_dict[chars[i]] = i

def make_one_hot(x):
	oh = [0]*28
	for xi in x:
		oh[char_dict[xi]] = 1
	return oh

for i in tqdm(range(len(corpus))):
	if corpus[i][0] == ' ':
		corpus[i] = corpus[i][1:31]
	else:
		corpus[i] = corpus[i][:30]
	#corpus[i] += '#'*(30 - len(corpus[i]))
	corpus[i] = make_one_hot(list(corpus[i])) #[make_one_hot(xx) for xx in corpus[i]]

X = np.array(corpus)
y = dataset.iloc[:, 2].values


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(y.reshape(-1,1)).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting classifier to the Training set
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
# classifier.fit(X_train, y_train)

# # Calculating y_pred
# y_pred = classifier.predict(X_test)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix, classification_report
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# TP = cm[1][1]
# FP = cm[0][1]
# TN = cm[0][0]
# FN = cm[1][0]
# TP, FP, TN, FN = float(TP), float(FP),float(TN),float(FN)
# Accuracy = (TP + TN)/(TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)
# F1_score = 2 * Precision * Recall / (Precision + Recall)

# print('Accuracy, Precision, Recall and F1 score respectively: ' + str(Accuracy) + ', ' + str(Precision) + ', ' + str(Recall) + ', ' + str(F1_score))
# print(classification_report(y_test, y_pred,target_names=classifier.classes_))


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, LSTM, Flatten
from keras.layers.normalization import BatchNormalization

classifier = Sequential()

classifier.add(Dense(input_dim=28, output_dim=32, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.1))  # Dropout overfitting
classifier.add(Dense(output_dim=32, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.1))
classifier.add(Dense(output_dim=16, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.1))

classifier.add(Dense(output_dim=4, activation='softmax'))


classifier.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ['accuracy']) # Nadam rmsprop
history = classifier.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, batch_size=128)


'''
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(LSTM(300, return_sequences=True, input_shape=(30,28)))
model.add(Dropout(0.2))
model.add(LSTM(300, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(output_dim=32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=2048)'''
