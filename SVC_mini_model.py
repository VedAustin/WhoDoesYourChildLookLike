# Program to build version 1 of the face model for Dad, Mom and Child. This model is used by Face_detection_adv_1

import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.externals import joblib
# Last col is the label
data = np.genfromtxt('D:/Documents/ImageClassification/Dad_Mom_Child.csv',delimiter=',',dtype=np.float32)

# SVC
X = data[:,:-1]
y = data[:,-1]
# Grid Search: define the parameter search space
parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000],'gamma': [0.01, 0.001, 0.0001]}
# search for the best classifier within the search space and return it
clf = GridSearchCV(SVC(), parameters).fit(X, y)
classifier = clf.best_estimator_

print clf.grid_scores_
print clf.best_params_

# Save model
model_name = 'face_detect_model_1.pkl'
_ = joblib.dump(classifier,model_name,compress=9)



# Check to see if the saved model produces same validaton score as original model
classifier2 = joblib.load(model_name)
avg_accuracy = np.mean(cross_validation.cross_val_score(classifier2,X,y,cv=10,scoring='accuracy'))
print 'Cross validaton score (kfold=10):%f' %(avg_accuracy)

