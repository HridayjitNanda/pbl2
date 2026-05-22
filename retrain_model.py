import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Adjust path if needed
DATA_PATH = 'clean_fakejobs.csv'

print('Loading data...')
data = pd.read_csv(DATA_PATH)

print('Preparing train/test split...')
X_train, X_test, y_train, y_test = train_test_split(data.text, data.fraudulent, test_size=0.3, random_state=42)

print('Fitting CountVectorizer...')
vect = CountVectorizer()
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)

print('Training RandomForestClassifier...')
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_dtm, y_train)

print('Saving model and vectorizer...')
with open('model.pkl', 'wb') as mf:
    pickle.dump(clf, mf)
with open('vectorizer.pkl', 'wb') as vf:
    pickle.dump(vect, vf)

print('Done. model.pkl and vectorizer.pkl are saved in the current directory.')
