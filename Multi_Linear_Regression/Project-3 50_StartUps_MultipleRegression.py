import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Importing the dataset using Panda
dataset = pd.read_csv('10.1 50_Startups.csv')

# Selecting Specific Rows and Columns for x and y:
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Converting Categorical data to meaningful data:
labelencoder = LabelEncoder()
x[:, 3] = labelencoder.fit_transform(x[:, 3])

#print first five rows
#print('labelled data===', x[0:5])

# Specify the column that needs to be replaced
oneEncoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [3])],   # The column numbers to be transformed (here is [3] but can be [0, 1, 2])
    remainder='passthrough'                                       # Leave the rest of the columns untouched
)

# Removing categorical column and replacing it with new columns:
x = oneEncode.fit_transform(x)
#print(x)

#Removing 1st extra column: Changing n to (n-1)
x = x[:, 1:]
#print(x.shape)

# Spliting the Database into Training and Test:
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#print(X_test.shape) # Use to check the dimention of a your array
#print(X_train.shape)

# Buliding Model using Linear-Regression Analysis:
regression = LinearRegression()
regression.fit(X_train, Y_train)

# Predicting "y" values of Test-sample:
Y_pred = regression.predict(X_test)
#print(Y_pred)
#print(Y_test)
#Y_pred = np.round(Y_pred[:], 2)
#print(Y_pred)

# Calculating R^2 Score, Coefficient and intercept:
coef = regression.coef_
#print(coef.shape)
print(coef)
print(regression.score(x, y))
print(regression.intercept_)
