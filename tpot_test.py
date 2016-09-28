# Using TPOT to determine the optimum classifier

# TPOT
from tpot import TPOT
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

# Last col is the label
data = np.genfromtxt('D:/Documents/ImageClassification/Dad_Mom_Child_large.csv',delimiter=',',dtype=np.float32)

# Filter out data with labels 0 (Dad) and 1 (Mom) as we are building our model on this.
data_model = data[np.where(data[:,-1] != 2.0)]

X_data = data_model[:,:-1]
y_data = data_model[:,-1]


tpot = TPOT(generations=40)
tpot.fit(X_data, y_data)
#print(tpot.score(X_test, y_test))
tpot.export('tpot_face_pipeline.py')

