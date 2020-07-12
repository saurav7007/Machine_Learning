import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

# IMPORTING OUR DATA:
dataset = pd.read_csv('7.1 Social_Network_Ads.csv')
#print(dataset)

# SELECTING SPECIFIC COLUMNS AND ROWS FOR X AND Y:
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values
#print(x)
#print(y)

# SPLITTING DATA-SET INTO TRAINING AND TEST:
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
#print(X_train)
#print(Y_train)
#print(X_train.shape) # Its displays total  no. of rows&columns present i.e 75% of 400 rows(whole dataset)
#print(Y_train.shape)

# NORMALIZING OUR X VALUES USING Z-SCORE:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
#print(X_train)
X_test = sc.transform(X_test)
#print(X_test)

# BUILDING A MODEL: USING K-NEIGHBOR
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric= 'minkowski')
classifier.fit(X_train, Y_train)

# PREDICTING VALUE OF Y:
Y_pred = classifier.predict(X_test)
#print(Y_pred)
#print(Y_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
#print(cm)
#print(cm[1, 1])
acc = (cm[0, 0] + cm[1, 1])/(cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
print(acc)
