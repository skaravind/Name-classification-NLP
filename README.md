# Name-classification-NLP

## Attempt One

To run the gender classifier (supports Python 3), `python3 gender_predictor.py`

and for race classifier, `python3 race_predictor.py`

### Model Reports

1. Gender Classification
* Model used: Logistic Regression with TFidfVectorization features.
* Data not resampled because oversampling caused worse overall performance
* Average metrics: 
  - Precision : 0.92      
  - Recall : 0.91    
  - F1 score : 0.90

2. Race Classification
* Model used: Logistic Regression with TFidfVectorization features.
* Data resampled using RandomOversampling as Hispanic data too low.
* Average metrics: 
  - Precision : 0.73    
  - Recall : 0.73    
  - F1 score : 0.73

## Attempt Two

I have trained a character-level as well as bigrams and trigrams embedding using gensim Word2vec and applied ANN based classification on the embeddings. All the code is in the IPython notebook `name_classifcation.ipynb`

1. Gender Classification - Val Acuracy (97.6%)

2. Race Classification - Val Accuracy (88%)

I am working on finding a better approach for solving problem 2.
