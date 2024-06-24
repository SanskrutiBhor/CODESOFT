#!/usr/bin/env python
# coding: utf-8

# # CodSoft Internship Task - Movie Rating Prediction with Python
# 

# ### Introduction
# * Movie Rating Prediction project involves building a model that predicts the rating of a movie based on features like genre, director, and actors. We use regression techniques to tackle this exciting problem. This enables us to explore data analysis, preprocessing, feature engineering, and machine learning modeling techniques 

# ### Goal
# * The main goal of this project is to analyze historical movie data and develop a model that accurately estimates the rating given to a movie by users or critics. By doing so, we aim to provide insights into the factors that influence movie ratings and create a model that can estimate the ratings of movies accurately.

# 
# ### Dataset: 
#   * Movie Rating Prediction Dataset

# ### Technologies Used
# * Python
# * Jupyter Notebook
# * Libraries: pandas, numpy, scikit-learn, seaborn, and matplotlib

# ## 1. Importing Libraries

# In[2]:


# import necessary libraries required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor


# ## 2. Reading Data

# In[11]:


# read the dataset into a dataframe
df = pd.read_csv(r"C:\Users\bhors\Desktop\Movies.csv",encoding='latin1')
# show first five records of dataframe
df.head()


# ## 3. Data Preprocessing

# In[12]:


# show the number of records and observations in the dataframe
df.shape


# In[13]:


# check out the information on the dataframe
df.info()


# In[14]:


# check out the missing values in each observation
df.isna().sum()


# In[29]:


# drop records with missing value in any of the following columns: Name, Year, Duration, Votes, Rating
df.dropna(subset=['Name', 'Year', 'Duration', 'Votes', 'Rating'], inplace=True)

# check the missing values in each observation again
df.isna().sum()


# In[35]:


# remove rows with duplicate movie records
df.drop_duplicates(subset=['Name', 'Year', 'Director'], keep='first', inplace=True)


# In[38]:


# remove commas from Votes column and convert to integer
df['Votes'] = df['Votes'].str.replace(',', '').astype(int)

# show the number of records and observations after cleaning the dataframe
df.shape


# In[39]:


# show the info on the cleaned dataframe
df.info()


# In[40]:


# show the statistics of the dataframe
df.describe()


# ## 4. Exploratory Data Analysis (EDA)

# #### i. Number of Movies each Year

# In[64]:


# group the data by Year and count the number of movies in each year
yearly_movie_counts = df['Year'].value_counts().sort_index()

# create a bar chart
plt.figure(figsize=(18, 9))
plt.bar(yearly_movie_counts.index, yearly_movie_counts.values, color='darkred')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.title('Number of Movies Released Each Year')
plt.show()


# #### ii. Creating Genre Dummy Columns and Analyzing Movie Counts by Genre

# In[60]:


# create dummy columns for each genre
dummies = df['Genre'].str.get_dummies(', ')
# creating a new dataframe which combines df and dummies
df_genre = pd.concat([df, dummies], axis=1)


# In[61]:


genre_columns = df_genre.columns[10:]  # Assuming genre columns start from the 11th column
genre_columns


# In[85]:


# group the data by genre_columns and count the number of movies in each genre
genre_movie_counts = df_genre[genre_columns].sum().sort_index()

# create a bar chart
plt.figure(figsize=(18, 9))
plt.bar(genre_movie_counts.index, genre_movie_counts.values, color='darkred')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.title('Number of Movies Released Per Genre')

plt.xticks(rotation=90)  
plt.show()


# #### iii. Top 20 Directors with the Most Movies

# In[67]:


# Analyzing count of movies of each director
director_movie_counts = df['Director'].value_counts()

# Create a bar chart
plt.figure(figsize=(10, 5))
bars = director_movie_counts.head(20).plot(kind='bar', color='maroon')
plt.xlabel('Director')
plt.ylabel('Number of Movies')
plt.title('Top 20 Directors with the Most Movies')
plt.xticks(rotation=90) 

# Add count labels on top of the bars
for bar in bars.patches:
    xval = bar.get_x() + bar.get_width() / 2
    yval = bar.get_height()
    plt.text(xval, yval, int(yval), ha='center', va='bottom')

plt.show()


# #### iv. Top 20 Actors with the most Movies

# In[68]:


# To Count Top 20 movies for each actor
actor_movie_counts = df['Actor 1'].value_counts()

# Create a bar chart
plt.figure(figsize=(10, 5))
actor_movie_counts.head(20).plot(kind='bar', color='maroon')
plt.xlabel('Actors')
plt.ylabel('Number of Movies')
plt.title('Top 20 Actors with the Most Movies')
plt.xticks(rotation=90) 

# Add count labels on top of the bars
for i, v in enumerate(actor_movie_counts.head(20)):
    plt.text(i, v, str(v), ha='center', va='bottom')

plt.show()


# #### v. Movie Duration vs. Rating Scatter Plot

# In[69]:


plt.figure(figsize=(20, 8))
# create a scatter plot with Duration and Rating relationship
sns.scatterplot(x=df['Duration'], y=df['Rating'],  color = 'maroon')
plt.xlabel('Duration of Movie (mins)')
plt.ylabel('Movie Rating')
plt.title('Movie Duration vs Rating')
plt.show()


# ## 5. Feature Engineering

# In[70]:


# dropping the columns from the dataframe since these are the least dependable observations for target variable 'Rating'
df.drop(['Name','Director','Actor 1','Actor 2','Actor 3'], axis=1,inplace=True)
# show first five records of the dataframe
df.head()


# In[72]:


# creating target variable and learning observations for the model
X = df[['Year','Duration','Votes']]
y = df['Rating']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=231)


# ## 6. Machine Learning Modeling Techniques

# #### i. Linear Regression Model

# In[73]:


# creating a liner regression model
lr = LinearRegression()

# training the data on linear regression model
lr.fit(X_train, y_train)

# predicting the test data on trained model
pred = lr.predict(X_test)

# evaluating linear regression model
r2_score(y_test,pred)


# #### ii. K-Nearest Neighbors (KNN) Regression Model

# In[74]:


# creating a range for number of neighbors parameter of the KNN model
kRange = range(1,40,1)

# creating an empty scores list
scores_list = []

# iterate every value in kRange list 
for i in kRange:
    # create a K Nearest Neighbor model with i as number of neighbors
    regressor_knn = KNeighborsRegressor(n_neighbors = i)
    
    # fit training data to the KNN model
    regressor_knn.fit(X_train,y_train)
    # evaluate the model
    pred = regressor_knn.predict(X_test)
    
    # append the regression score for evaluation of the model to scores_list
    scores_list.append(r2_score(y_test,pred))


# In[75]:


plt.figure(figsize=(12,8))
# create a line graph for showing regression score (scores_list) for respective number of neighbors used in the KNN model
plt.plot(kRange, scores_list, linewidth=2, color='green')
# values for x-axis should be the number of neighbors stored in kRange
plt.xticks(kRange)
plt.xlabel('Neighbor Number')
plt.ylabel('r2_Score of KNN')
plt.show() 


# In[76]:


# Creating a KNN model with best parameters i.e., number of neighbors = 23
regressor_knn = KNeighborsRegressor(n_neighbors = 23)

# fit training data to the KNN model
regressor_knn.fit(X_train,y_train)
# evaluate test data on the model
pred = regressor_knn.predict(X_test)
# show regression score
r2_score(y_test,pred)


# #### iii. SGD Regression

# In[77]:


from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score

# Create an instance of the SGDRegressor
sgd_regressor = SGDRegressor(max_iter=100, random_state=1)  # You can adjust the max_iter and random_state

# Fit the model to your training data
sgd_regressor.fit(X_train, y_train)

# Make predictions
pred = sgd_regressor.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, pred)

print("R-squared score:", r2)


# #### iv. Random Forest Regression

# In[78]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=1)
rf_regressor.fit(X_train, y_train)
rf_pred = rf_regressor.predict(X_test)
r2_rf = r2_score(y_test, rf_pred)
print(f'R-squared score (Random Forest): {r2_rf}')


# #### v. Gradient Boosting Regression

# In[80]:


from sklearn.ensemble import GradientBoostingRegressor
gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=231)
gb_regressor.fit(X_train, y_train)
gb_pred = gb_regressor.predict(X_test)
r2_gb = r2_score(y_test, gb_pred)
print(f'R-squared score: {r2_gb}')


# ### Conclusion
# * In conclusion, this project allowed us to explore various data analysis, data preprocessing, feature engineering, and machine learning modeling techniques. It provided valuable insights into the factors influencing movie ratings and equipped us with a model for accurate movie rating predictions.

# In[ ]:




