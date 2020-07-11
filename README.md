# Multi-Linear Regression Analysis using Machine_Learning:
We want to analyse the given data on StartUps from different States of America using **Multi-Linear Regression Model**.
Here, We are using following packages for dat- pre-processing and analysis:
1. numpy: helps us to work with array.
2. pandas: helps us to import data available in different file format
3. matplotlib: helps us to plot different graphs like (sactter-plot)
4. sklearn: helps to preprocess the data and analysis it.

Among all these libraries **sklearn** is most important. It has inbulit functions which are very useful whiles working with Machine Learning.

### sklearn.preprocessing:
Generally during **Multi-Linear Regression Analysis**, *Categorical data-types* (dataframe with strings values) might create problems during analysis.
Hence, we use **Sklearn** library for preprocessing those data and transforming them into more meaningful information that our computer can understand.
Here, we are going to use class preprocessing from sklearn library to use two functions:
- LabelEncoder: will help is to convert unique values present in ***categorical column*** to unique-numerical values (***n***).
- OneHotEncoder: will help us to _**remove** the categorical cloumn and **replace** it with ***n*** numbers of new cloumns eaching with an array of *Boolean values* i.e. O or 1.

### sklearn.model_selection:
Model_selection has **train_test_split** function which helps to divide the dataset into training and test set on the basis of test_size/train_size.

### sklearn.linear_model:
This class has function to calcualte **Linear-Regression** and then addiction methods to predict values of ***y*** for given **Test-dataset**. It can also be used to
calcuate the value R<sup>2</sup>, coefficient, and intercept.
