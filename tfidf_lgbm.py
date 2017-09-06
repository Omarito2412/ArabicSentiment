from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import pandas as pd
from nltk.corpus import stopwords
from lightgbm import LGBMClassifier
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


params = {
	"max_depth": [5, 6, 7],
	"subsample": [0.99, 1],
	"n_estimators": [100]
}
grid = GridSearchCV(LGBMClassifier(), params, n_jobs=4, verbose=0)
grid.fit(X_Train_tfidf, Y_Train)

lgbm = grid.best_estimator_
Y_Pred = lgbm.predict(X_Test_tfidf)
print("LGBM Accuracy: ")
print(float(np.sum(Y_Test.values == Y_Pred)) / len(Y_Pred))

probabilities = pd.DataFrame(lgbm.predict_proba(X_Test_tfidf), columns=lgbm.classes)
probabilities.to_csv("lgbm_probabilities.csv", index=False)