from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
import numpy as np

stop_words = pd.Series(stopwords.words('arabic'))
stop_words = stop_words.apply(lambda x: x.encode('utf-8'))

d_train = pd.read_csv("train.csv").dropna()
d_test = pd.read_csv("test.csv").dropna()

X_Train = d_train.drop('class', axis=1)
Y_Train = d_train['class']

X_Test = d_test.drop('1', axis=1)
Y_Test = d_test['1']

vectorizer = TfidfVectorizer(max_df = 0.8, use_idf =True,
	stop_words = stop_words.tolist(), analyzer='word', lowercase=False, sublinear_tf=True)
X_Train_tfidf = vectorizer.fit_transform(X_Train.values.ravel())
X_Test_tfidf = vectorizer.transform(X_Test.values.ravel())

nb = MultinomialNB()
nb.fit(X_Train_tfidf, Y_Train)
Y_Pred = nb.predict(X_Test_tfidf)
print("Naive Bayes Accuracy: ")
print(float(np.sum(Y_Test.values == Y_Pred)) / len(Y_Pred))

probabilities = pd.DataFrame(nb.predict_proba(X_Test_tfidf), columns=nb.classes_)
probabilities.to_csv("nb_probabilities.csv", index=False)