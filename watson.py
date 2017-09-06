from watson_developer_cloud import NaturalLanguageClassifierV1
import pandas as pd
from tqdm import tqdm

natural_language_classifier = NaturalLanguageClassifierV1(
  username='INSERT YOURS',
  password='INSERT YOURS')

CLASSIFIER_ID = 'INSERT YOURS'

y_pred = pd.DataFrame(columns=['NEG', 'OBJ', 'POS', 'NEUTRAL', 'TOP'])

test = pd.read_csv("d_test.csv", header=None)
test = test.dropna()

for item in tqdm(test.iterrows()):
	classes = natural_language_classifier.classify(CLASSIFIER_ID, item[1][0])
	temp = {}
	for class_item in classes['classes']:
	    temp[class_item['class_name']] = class_item['confidence']
	temp['TOP'] = classes['top_class']
	y_pred.loc[item[0]] = temp

	
y_pred.to_csv("y_pred_watson.csv")
