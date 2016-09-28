# Using SVC to classify Shrimayi's images as either Rachna or Vedraj

import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

# Last col is the label
data = np.genfromtxt('D:/Documents/ImageClassification/Ved_Rach_Shim_large.csv',delimiter=',',dtype=np.float32)

# Filter out data with labels 0 (Vedraj) and 1 (Rachna) as we are building our model on this.
data_model = data[np.where(data[:,-1] != 2.0)]
data_test = data[np.where(data[:,-1] == 2.0)]
data_test = data_test[:,:-1]


X = data_model[:,:-1]
y = data_model[:,-1]

##########
# SVC
# Grid Search: define the parameter search space
parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000],'gamma': [0.01, 0.001, 0.0001]}
# search for the best classifier within the search space and return it
clf = GridSearchCV(SVC(), parameters).fit(X, y)
#print clf.best_params_
# Cross Validation
classifier_cv = SVC().set_params(**clf.best_params_)
avg_accuracy_svc = np.mean(cross_validation.cross_val_score(classifier_cv,X,y,cv=10,scoring='accuracy'))
print 'Cross validaton score (SVC,kfold=10):%f' %(avg_accuracy_svc)
##########
# Logistic Regression: Refer topt_test.py
from sklearn.linear_model import LogisticRegression

logR = LogisticRegression(C=100)

# Cross Validation
avg_accuracy_log = np.mean(cross_validation.cross_val_score(logR,X,y,cv=10,scoring='accuracy'))
print 'Cross validaton score (LogisticRegression,kfold=10):%f' %(avg_accuracy_log)

if avg_accuracy_log >= avg_accuracy_svc:
	classifier = logR.fit(X,y)
else:
	classifier = clf.best_estimator_

print classifier
predicted_labels = classifier.predict(data_test)
rachna_count = sum(predicted_labels==1)
vedraj_count = sum(predicted_labels==0)
print "Number of Shrimayi's photos:%d" %(data_test.shape[0])
print "Score for Rachna:%f,\nScore for Vedraj:%f"%(rachna_count*1./data_test.shape[0],vedraj_count*1./data_test.shape[0])