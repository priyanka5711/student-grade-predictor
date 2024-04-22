import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

plt.style.use("ggplot")
pd.options.display.max_columns = 50

# Load the dataset
df = pd.read_csv("student-mat.csv", sep=";")

# Exploratory Data Analysis

# Check the shape of the dataset
print("Shape of the dataset:", df.shape)

# Take a quick look at the dataset
print("First few rows of the dataset:")
print(df.head())

# Summary statistics
print("Summary statistics of the dataset:")
print(df.describe())

# Creating subplots for categorical variables
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 10))

# Plotting countplots for categorical variables
sns.countplot(x='school', data=df, ax=axes[0, 0])
sns.countplot(x='sex', data=df, ax=axes[0, 1])
sns.countplot(x='address', data=df, ax=axes[0, 2])
sns.countplot(x='famsize', data=df, ax=axes[0, 3])
sns.countplot(x='Pstatus', data=df, ax=axes[1, 0])
sns.countplot(x='schoolsup', data=df, ax=axes[1, 1])
sns.countplot(x='famsup', data=df, ax=axes[1, 2])
sns.countplot(x='paid', data=df, ax=axes[1, 3])

# Adding titles to the subplots
axes[0, 0].set_title('School')
axes[0, 1].set_title('Sex')
axes[0, 2].set_title('Address')
axes[0, 3].set_title('Family Size')
axes[1, 0].set_title('Parent Cohabitation Status')
axes[1, 1].set_title('Extra Educational Support')
axes[1, 2].set_title('Family Educational Support')
axes[1, 3].set_title('Extra Paid Classes')

# Adjusting layout
plt.tight_layout()
plt.show()

# Histogram of age distribution
sns.histplot(data=df, x='age', bins=10, discrete=True)
plt.title("Age Distribution")
plt.show()

# Boxplot of age distribution by sex
sns.boxplot(x='sex', y='age', data=df)
plt.title("Age Distribution by Sex")
plt.show()

# Barplot comparing studytime by school
sns.barplot(x='school', y='studytime', data=df)
plt.title("Average Study Time by School")
plt.show()

# Scatterplot comparing G1 and G2 grades
sns.scatterplot(x='G1', y='G2', data=df)
plt.title("Scatterplot of G1 vs G2 Grades")
plt.show()

# Multivariate Analysis

# Pairplot to visualize relationships between multiple numeric variables
sns.pairplot(df[['studytime', 'G1', 'G2', 'G3']])
plt.title("Pairplot of Study Time, G1, G2, and G3 Grades")
plt.tight_layout()
plt.show()

# Median score on each exam from each school
print("Median score on each exam from each school:")
print(df.groupby("school")[["G1", "G2", "G3"]].median())

# Preparing the dataset

# One-hot encoding with customized column names
encoded_df = pd.get_dummies(df, columns=['sex'], prefix=['Is'], dtype="int")

# Subsetting the dataset
work_df = encoded_df.iloc[:300].drop(columns=["school"])

# One-hot encoding with additional features
encoded_df = pd.get_dummies(work_df, columns=['Fedu', 'Medu', 'studytime', 'higher', 'activities'], dtype="int")

# Selecting features and target variable
y = encoded_df["G3"].iloc[:240]
features = ['age', 'Fedu_0', 'Fedu_1', 'Fedu_2', 'Fedu_3', 'Fedu_4', 'Medu_0', 'Medu_1', 'Medu_2', 'Medu_3', 'Medu_4', 'studytime_1', 'studytime_2', 'studytime_3', 'studytime_4', 'higher_no', 'higher_yes', 'activities_no', 'activities_yes', "G1", "G2"]
X = encoded_df[features].iloc[:240]
test_X = encoded_df[features].iloc[241:]

# Define and train the model
testScoreModel = DecisionTreeRegressor(random_state=1)
testScoreModel.fit(X, y)

# Using the model

# Predict grades for the last five students
print("Predicted grades:", [int(x) for x in list(testScoreModel.predict(test_X.tail()))], sep="\n")
print("Actual grades:", list(encoded_df["G3"].tail()), sep="\n")