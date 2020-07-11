import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv('10.1 50_Startups.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Converting Categorical data to meaningful data:

labelencoder = LabelEncoder()
x[:, 3] = labelencoder.fit_transform(x[:, 3])

# print first five rows
print('labelled data===', x[0:5])

# specify the column that needs to be replaced
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [3])],   # The column numbers to be transformed (here is [3] but can be [0, 1, 2])
    remainder='passthrough'                                       # Leave the rest of the columns untouched
)

# Removing categorical column and replacing it with new columns:
x = ct.fit_transform(x)

# print first five rows
print('transformed data===', x[0:5])






# Removing 1st extra column: Changing n to (n-1)

