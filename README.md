# Name-classification-NLP

To run the gender classifier (supports Python 3), `python3 gender_predictor.py`

and for race classifier, `python3 race_predictor.py`

## Model Reports

1. Gender Classification
* Model used: Logistic Regression with TFidfVectorization features.
* Data not resampled
* Average metrics: 
  - Precision : 0.92      
  - Recall : 0.91    
  - F1 score : 0.90

2. Race Classification
* Model used: Logistic Regression with TFidfVectorization features.
* Data not resampled using RandomOversampling as Hispanic data too low.
* Average metrics: 
  - Precision : 0.73    
  - Recall : 0.73    
  - F1 score : 0.73
