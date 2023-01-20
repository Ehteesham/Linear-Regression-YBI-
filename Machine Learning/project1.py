# Project: Linear Regression To Predict the Cement Compressive Strength

# importing all the necessary library
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing our data set
df = pd.read_csv(
    "https://github.com/ybifoundation/Dataset/raw/main/Concrete%20Compressive%20Strength.csv")

#  About DataSet
# 1. 9 columns are there.
# 2. There is no null values are present in any columns
# 3. Total Number of rows are 1030
# 4. There is no object data type
# 5. Shape of the DataSet is (1035, 9)


# More About DataSet

# Head will print First 5 rows of all columns
df_head = df.head()
# print(df_head)


# Now we'll check for the columns name
column_name = df.columns
# print(column_name)


# Now we'll check how many rows are there and thus any column contain the null value or not
df_info = df.info()
# print(df_info)

# Now we'll check the statistical summary
df_stats = df.describe()
# print(df_stats)

# Now we'll specialy check is there any missing value is present in the columns or not
df_null = df.isnull().sum()
# print(df_null)


# Now we have to check how many category are in each columns
df_cat = df.nunique()
# print(df_cat)


# Now we'll plot the Pair grid plot using scatter plot of each data in the dataset
pair_grid = sns.PairGrid(df)
pair_grid.map(plt.scatter)
# plt.show()

#  you can also see the specific column value in category
df_spec_col = df['Age (day)'].value_counts()
# print(df_spec_col)


# First we'll rename our dependable variable column name
df.rename(columns={
          'Concrete Compressive Strength(MPa, megapascals) ': 'Concrete'}, inplace=True)
# print(df.columns)

# Now we'll start Building our model from here

# We'll assign the depandable(y) and independable variable(X)
y = df['Concrete']

X = df[['Cement (kg in a m^3 mixture)',
       'Blast Furnace Slag (kg in a m^3 mixture)',
        'Fly Ash (kg in a m^3 mixture)', 'Water (kg in a m^3 mixture)',
        'Superplasticizer (kg in a m^3 mixture)',
        'Coarse Aggregate (kg in a m^3 mixture)',
        'Fine Aggregate (kg in a m^3 mixture)', 'Age (day)']]


#  Now we'll split the dataset into two parts
#  One is for training and other is for testing


# Train = 70% and Test = 30% data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=2529)

# Now Verify the Shape of all this

df_shape = X_train.shape, X_test.shape, y_train.shape, y_test.shape
# print(df_shape)

# Selecting the model

# Now making an object for class
reg = LinearRegression()

# Training the Model
reg.fit(X_train, y_train)

# prediction start
y_pred = reg.predict(X_test)
# print(y_pred)

# Coefficent and intercept
coeff = reg.coef_   # It'll be 8 values are present in coeff
# print(coeff)

intercept = reg.intercept_    # It'll be only one value present in intercept
# print(intercept)

# Now checking the model accuracy


# MAE
mae_acc = mean_absolute_error(y_test, y_pred)
# print(mae_acc)

# MAPE
mape_acc = mean_absolute_percentage_error(y_test, y_pred)
# print(mape_acc)

# MSE
mse_acc = mean_squared_error(y_test, y_pred)
# print(mse_acc)


# Now we'll predict by giving some values which is out of dataset

# Here we have selected our row by index number
new_df = df.iloc[[405]]
# print(new_df)


# Now we'll drop the dependable column from the dataset
X_new = new_df.drop(['Concrete'], axis = 1)
# print(X_new)

# Now we'll start predicting
Y_pred = reg.predict(X_new)
# print(Y_pred)

# Here our Model is fully ready to predict 