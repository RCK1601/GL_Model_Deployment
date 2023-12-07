#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# The Iris dataset is a classic dataset for classification, machine learning, and data visualization.
# 
# The dataset contains: 3 classes (different Iris species) with 50 samples each, and then four numeric properties about those classes: Sepal Length, Sepal Width, Petal Length, and Petal Width.
# 
# One species, Iris Setosa, is "linearly separable" from the other two. This means that we can draw a line (or a hyperplane in higher-dimensional spaces) between Iris Setosa samples and samples corresponding to the other two species.
# 
# Predicted Attribute: Different Species of Iris plant.
# 
# 

# ## Objective
# 
# The objective of this project was to gain introductory exposure to Machine Learning Classification concepts along with data visualization. The project makes heavy use of Scikit-Learn, Pandas and Data Visualization Libraries.

# ## Iris Data set
# 
# **Iris flower is divided into 3 species:**
# - setosa
# - versicolor
# - virginica
# 
# **The iris dataset consists of 4 features:**
# - Sepal Length
# - Sepal Width
# - Petal Length
# - Petal Width

# In[1]:


# To help with reading and manipulating data
import pandas as pd
import numpy as np

# To help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns


# To split the data
from sklearn.model_selection import train_test_split

# To help with model building
from sklearn.linear_model import LogisticRegression

# To help with feature scaling
from sklearn.preprocessing import StandardScaler

# Ignore all warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Loading the dataset

df = pd.read_csv('Iris.csv')


# ### Data Overview

# In[3]:


# First 5 rows of the data set

df.head()


# In[4]:


# Last 5 rows of the data set

df.tail()


# In[5]:


# Checking the number of rows and columns in the train data

df.shape


# There are 150 rows and 6 columns in the Iris data set.

# In[6]:


# Check the data types of the columns in the dataset

df.info()


# In[7]:


# Check for the missing values in the data

df.isnull().sum()


# There are no missing values in the data set.

# In[8]:


# Check for duplicate values in the data

df.duplicated().sum()


# There are no duplicated rows in the data set.

# ## Exploratory Data Analysis (EDA)

# In[9]:


# Statistical summary of the numerical columns in the data

df.describe().T


# In[10]:


# Check the distribution of the species

df["Species"].value_counts()


# ### Sepal Length vs Sepal Width

# In[11]:


# Scatter plot of the two features SepalLengthCm and SepalWidthCm
sns.FacetGrid(df, hue="Species", height=6) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
   .add_legend()
plt.title('Sepal Length Vs Sepel Width')
plt.show()


# **Observations:**
# 1. Using SepalLengthCm and SepalWidthCm features, we can distinguish Iris-setosa flowers from others.
# 2. Seperating Iris-versicolor from Iris-viginica is much harder as they have considerable overlap.

# ### Petal Length vs Petal Width

# In[12]:


# Scatter plot of the two features PetalLengthCm and PetalWidthCm
sns.FacetGrid(df, hue="Species", height=6) \
   .map(plt.scatter, "PetalLengthCm", "PetalWidthCm") \
   .add_legend()
plt.title('Petal Length Vs Petal Width')
plt.show()


# **Observations:**
# 1. Using PetalLengthCm and PetalWidthCm features, we can distinguish Iris-setosa flowers from others.
# 2. PetalLengthCm and PetalWidthCm features selection is giving better results than SepalLengthCm and SepalWidthCm features selection.

# ### Pair plot

# In[13]:


# Ploting a pair plot to explore all the columns
sns.pairplot(df, hue="Species", height=3);
plt.show()


# **Observations:**
# 1. PetalLengthCm and PetalWidthCm are the best features to identify various flower types.
# 2. Iris-setosa can be easily identified
# 3. Iris-virginica and Iris-versicolor have some overlap.

# In[14]:


# Boxplot to explore all the columns in the data set
plt.figure(figsize=(15,10))

# Boxplot of SepalLengthCm
plt.subplot(2,2,1)
sns.boxplot(x='Species', y = 'SepalLengthCm', data=df)

# Boxplot of SepalWidthCm
plt.subplot(2,2,2)
sns.boxplot(x='Species', y = 'SepalWidthCm', data=df)

# Boxplot of PetalLengthCm
plt.subplot(2,2,3)
sns.boxplot(x='Species', y = 'PetalLengthCm', data=df)

# Boxplot of PetalWidthCm
plt.subplot(2,2,4)
sns.boxplot(x='Species', y = 'PetalWidthCm', data=df)


# In[15]:


# Violinplot to explore all the columns in the data set
plt.figure(figsize=(15,10))

# Violinplot of SepalLengthCm
plt.subplot(2,2,1)
sns.violinplot(x='Species', y = 'SepalLengthCm', data=df)

# Violinplot of SepalWidthCm
plt.subplot(2,2,2)
sns.violinplot(x='Species', y = 'SepalWidthCm', data=df)

# Violinplot of PetalLengthCm
plt.subplot(2,2,3)
sns.violinplot(x='Species', y = 'PetalLengthCm', data=df)

# Violinplot of PetalWidthCm
plt.subplot(2,2,4)
sns.violinplot(x='Species', y = 'PetalWidthCm', data=df)


# ### Data Pre-processing

# In[16]:


# Removing the ID Column

df.drop("Id", axis=1, inplace = True)


# In[17]:


# Separating Independent and Dependent Varibles

X = df.iloc[:, 0: 4].values
y = df.iloc[:, 4].values


# In[18]:


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[19]:


# Data Overview of the X, y dataframes

print("The shape of the X_train is:", X_train.shape)
print("The shape of the X_test is:", X_test.shape)

print("The shape of the y_train is:", y_train.shape)
print("The shape of the y_test is:", y_test.shape)


# ### Feature Engineering

# In[20]:


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ### Logistic Regression

# In[21]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 42)
lr.fit(X_train, y_train)


# In[22]:


# Predicting the Test set results
y_pred = lr.predict(X_test)

y_pred


# In[23]:


# Measuring Accuracy
from sklearn import metrics
print('The accuracy of Logistic Regression is: ', metrics.accuracy_score(y_pred, y_test))


# In[24]:


lr.predict([[9,8,5,6]])


# ### Saving the Model

# In[25]:


# Importing necessary libraries
from joblib import dump, load


# In[26]:


# Saving my model with the name my_linear_regression_model
dump(lr, 'my_linear_regression_model.joblib')


# ### Loading the Model

# In[27]:


from joblib import dump, load
my_lr_model = load('my_linear_regression_model.joblib') 

my_lr_model


# ### Making Predictions

# In[28]:


# Making a single prediction
my_lr_model.predict([[5,6,7,8]])


# In[29]:


# Making a multiple predictions
my_lr_model.predict([[5.6,6.6,7.6,8.6],[6.8,7.8,8.8,9.8],[4.3,5.4,6,5],[5.3,4.7,8,9.5]])


# ### Making a function to give predictions

# In[39]:


from joblib import dump, load

def load_model_predict(model_path, data_points):

    # Loading the model
    my_lr_model = load(model_path) 

    # Creating a DataFrame 
    temp_dict = {'sepal_length':data_points[0], 'sepal_width':data_points[1], 'petal_length':data_points[2], 'petal_width':data_points[3]}
    df_temp = pd.DataFrame(temp_dict)

    # Predicting on my_data
    predictions = my_lr_model.predict(df_temp)

    return predictions


# In[40]:


# For making 1 prediction
load_model_predict('my_linear_regression_model.joblib', [[5.1],[3.5],[1.4],[0.2]])


# In[41]:


# For making multiple predictions
load_model_predict('my_linear_regression_model.joblib', [[5.1,7,8, 4.3],[3.5,9,8, 3.0],[1.4,5,4, 1.1],[0.2,2,4, 0.1]])


# In[ ]:




