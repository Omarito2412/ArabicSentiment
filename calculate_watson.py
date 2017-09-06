import numpy as np
import pandas as pd

d_test = pd.read_csv("test.csv").dropna()
Y_Test = d_test['1']
Y_Watson = pd.read_csv("y_pred_watson.csv", index_col=0)

print("Watson Accuracy: ")
print(float(np.sum(Y_Test.values == Y_Watson['TOP'])) / len(Y_Watson))
