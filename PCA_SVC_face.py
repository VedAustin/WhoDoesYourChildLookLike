# Using a combination of PCA (for dimensionality reduction purposes) and SVC to classify images
# This is to visually test if there are distinguishable features between mom and dad. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

# Last col is the label
data = np.genfromtxt('D:/Documents/ImageClassification/Dad_Mom_Child.csv',delimiter=',',dtype=np.float32)

# Filter out data with labels 0 (Dad) and 1 (Mom) as we are building our model on this.
data_model = data[np.where(data[:,-1] != 2)]

x_train,x_test,y_train,y_test = train_test_split(data_model[:,:-1],data_model[:,-1],test_size=0,random_state=0)

# PCA (visual test)
n_comp = 2
pca = RandomizedPCA(n_components=n_comp)
X = pca.fit_transform(x_train)

df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label":np.where(y_train==1., "Mom", "Dad")})
colors = ["red", "blue"]
for label, color in zip(df['label'].unique(), colors):
    mask = df['label']==label
    plt.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
plt.legend()
plt.show()
'''
# PCA (thorough test): if the number of columns in the data set >> 500, use PCA else use just SVC
n_comp = 150
pca = RandomizedPCA(n_components=n_comp).fit(x_train)
train_X = pca.transform(x_train)
test_X = pca.transform(x_test)

# SVC
# Grid Search: define the parameter search space
train_X = x_train
test_X = x_test
parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000],'gamma': [0.01, 0.001, 0.0001]}
# search for the best classifier within the search space and return it
clf = GridSearchCV(SVC(), parameters).fit(train_X, y_train)
classifier = clf.best_estimator_
#print classifier

predicted_labels = classifier.predict(test_X)
print classification_report(y_test,predicted_labels)
print pd.crosstab(y_test,predicted_labels, rownames=['actual'], colnames=['predicted'])
'''