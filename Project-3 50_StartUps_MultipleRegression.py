import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('10.1 50_Startups.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Converting Categorical data to meaningful data:
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:, 3] = labelencoder.fit_transform(x[:, 3])
#print(x)

# Removing categorical column and replacing it with new columns:
oneEncode = OneHotEncoder(categorical_features= [3])
x = oneEncode.fit_transform(x).toarray()

# Removing 1st extra column: Changing n to (n-1)

