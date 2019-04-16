import pandas as pd
import gensim 
from gensim.models import Word2Vec 
from tqdm import tqdm
import re
import numpy as np
  
dataset = pd.read_csv("data/names_combined.csv")

dataset.dropna(inplace=True)
dataset.replace('b','black',inplace=True)
  
data = [] 
corpus = []

def features(name):
	name = re.sub(' ','',name)
	bigrams = []
	trigrams = []
	for i in range(len(name)-1):
		bigrams.append(name[i:i+2])
		if i < len(name) - 2:
			trigrams.append(name[i:i+3])
	trigrams.append(name[-3:])
	return bigrams

for i in tqdm(range(len(dataset['name']))):
	temp = []
	review = re.sub('[^a-zA-Z]', ' ', dataset['name'].values[i])
	review = review.lower()
	temp = list(review) + features(review)
	data.append(temp)
	corpus.append(review)

  
# Create CBOW model 
model1 = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5) 
print('model created')
model1.train(data,total_examples=model1.corpus_count, epochs=25)
print('model trained')

X = []
for c in tqdm(corpus):
	vecs = model1[list(c) + features(c)]
	X.append(vecs.sum(axis=0))
X = np.array(X)

y = dataset.iloc[:, 2].values
from imblearn.over_sampling import SMOTE, RandomOverSampler
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)

# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder()
# y = enc.fit_transform(y.reshape(-1,1)).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting classifier to the Training set
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, verbose=2, n_jobs=-1)
# classifier.fit(X_train, y_train)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, LSTM, Flatten
from keras.layers.normalization import BatchNormalization

classifier = Sequential()

classifier.add(Dense(input_dim=100, output_dim=128, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.1))  # Dropout overfitting
classifier.add(Dense(output_dim=64, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.1))
classifier.add(Dense(output_dim=32, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.1))

classifier.add(Dense(output_dim=4, activation='softmax'))


classifier.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ['accuracy']) # Nadam rmsprop
history = classifier.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, batch_size=128)

'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Calculating y_pred
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)

TP = cm[1][1]
FP = cm[0][1]
TN = cm[0][0]
FN = cm[1][0]
TP, FP, TN, FN = float(TP), float(FP),float(TN),float(FN)
Accuracy = (TP + TN)/(TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_score = 2 * Precision * Recall / (Precision + Recall)

print('Accuracy, Precision, Recall and F1 score respectively: ' + str(Accuracy) + ', ' + str(Precision) + ', ' + str(Recall) + ', ' + str(F1_score))
print(classification_report(y_test, y_pred,target_names=classifier.classes_))'''