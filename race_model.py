# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle


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


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#cv = CountVectorizer(max_features = 1200)
cv = TfidfVectorizer(max_features = 1000, ngram_range=(1, 3))
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 2].values

print('vectorized')

ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)
print('random oversampled. Now fitting classifier')

filename = 'corpus.sav'
pickle.dump(cv, open(filename, 'wb'))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting classifier to the Training set
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, verbose=2, n_jobs=-1)
# classifier.fit(X_train, y_train)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=200, random_state = 0, solver='lbfgs', multi_class='multinomial', verbose=2, n_jobs=3, class_weight={'indian':1,'black':1.5,'hispanic':1.5,'white':2})
classifier.fit(X_train, y_train)
# from sklearn.naive_bayes import MultinomialNB
# classifier = MultinomialNB()
# classifier.fit(X_train, y_train)

# Calculating y_pred
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)

print(cm)

print(classification_report(y_test, y_pred,target_names=classifier.classes_))

# save the model to disk
filename = 'race_model.sav'
pickle.dump(classifier, open(filename, 'wb'))