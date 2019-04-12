import pickle

classifier = pickle.load(open('models/race_model.sav', 'rb'))
cv = pickle.load(open('data/corpus.sav', 'rb'))

while True:
    name = input('Input Name:')
    name = cv.transform([name]).toarray()
    predval = classifier.predict(name)
    predprob = classifier.predict_proba(name)
    print('Race: '+predval[0])
    classes = classifier.classes_
    print('Confidences...')
    for i in range(len(classes)):
    	print(f'{classes[i]}: {predprob[0][i]*100}%')