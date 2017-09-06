import pandas as pd
import numpy as np

lexicon = pd.read_csv("NileULex.csv", header=None)

mapper = {'compound_neg': -1, 'negative': -1, 'compound_pos': 1, 'positive': 1}

lexicon['scores'] = lexicon[1].map(mapper)


d_test = pd.read_csv("test.csv").dropna()

# Because the lexicon is only constrained to POS/NEG, I'm removing OBJ
d_test = d_test.drop(d_test[d_test['1'] == 'OBJ'].index)
X_Test = d_test.drop('1', axis=1)
Y_Test = d_test['1']

word2senti = dict(zip(lexicon[0], lexicon['scores']))

def evaluate(tweet):
	score = 0
	for token in tweet.split(" "):
		if(token in word2senti):
			score += word2senti[token]
	if(score > 0):
		return "POS"
	elif(score < 0):
		return "NEG"
	return "NEUTRAL"

Y_Pred = X_Test['0'].apply(evaluate)

print("Lexicon only accuracy: ")
print(float(np.sum(Y_Test.values == Y_Pred)) / len(Y_Pred))
