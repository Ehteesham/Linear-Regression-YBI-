# Importing All the Necessary Library

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Importing our Data Set using pandas

df = pd.read_csv(
    "https://github.com/ybifoundation/Dataset/raw/main/MultipleDiseasePrediction.csv")

# About Our Data Set
# 1. 133 Columns
# 2. 4920 rows
# 3. This dataset is about a person have having a Multiple Disease

# Printing the first five Row of the Dataset
df_head = df.head()
# print(df_head)

# Info of the Data Set
df_info = df.info()
# print(df_info)

# Summary Statistic
df_stats = df.describe()
# print(df_stats)


df_null = df.isnull().sum()
# print(df_null)     # There is no Null Values


# Total number of Unique category in columns
df_cat = df.nunique()
# print(df_cat)

# correlation
df_cor = df.corr()
# print(df_cor)

# Pair plot
# g = sns.PairGrid(df)
# g.map(plt.scatter)
# plt.show()

df_column = df.columns
# print(df_column)

# define y and x
y = df['prognosis']
X = df.drop(['prognosis'], axis=1)

# Now train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=2529)

# shape
df_shape = X_train.shape, X_test.shape, y_train.shape, y_test.shape
# print(df_shape)

# select model

clas = RandomForestClassifier()

# train model
clas.fit(X_test, y_test)

# predict with model
y_pred = clas.predict(X_test)
# print(y_pred)

# Accuracy
model_accuracy = accuracy_score(y_test, y_pred)
# print(model_accuracy)

conf_mat = confusion_matrix(y_test, y_pred)
# print(conf_mat)

# Classification report
class_report = classification_report(y_test, y_pred)
# print(class_report)