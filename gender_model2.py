# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle
from collections import Counter
import sklearn_crfsuite

from imblearn.over_sampling import SMOTE, RandomOverSampler

# Importing the dataset
dataset = pd.read_csv("data/names_combined.csv")

dataset.dropna(inplace=True)
dataset.replace('b','black',inplace=True)

# Cleaning the texts
import re
import nltk
corpus = []

def features(name):
	name = re.sub(' ','',name)
	fets = {1:name[0],2:name[:1],3:name[:2],4:name[-1],5:name[-2:],6:name[-3:],7:str(len(name))}
	return fets

drop = []

for i in tqdm(range(len(dataset['name']))):
    review = re.sub('[^a-zA-Z]', ' ', dataset['name'].values[i])
    review = review.lower()
    try:
    	corpus.append(features(review))
    except:
    	drop.append(i)


# # Creating the Bag of Words model
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# #cv = CountVectorizer(max_features = 1200)
# cv = TfidfVectorizer(max_features = 1200, ngram_range=(1, 3))
# X = cv.fit_transform(corpus).toarray()
dataset.drop(drop, inplace=True)

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)

X = dv.fit_transform(corpus)
y = dataset.iloc[:, 2].values

print(X.shape)

print('vectorized')

# ros = RandomOverSampler(random_state=42)
# X, y = ros.fit_resample(X, y)

# print('random oversampled. Now fitting classifier')

# filename = 'corpus.sav'
# pickle.dump(cv, open(filename, 'wb'))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, verbose=2, n_jobs=-1)
classifier.fit(X_train, y_train)
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
# classifier.fit(X_train, y_train)


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
print(classification_report(y_test, y_pred,target_names=classifier.classes_))
# save the model to disk
filename = 'gender_model2.sav'
pickle.dump(classifier, open(filename, 'wb'))
