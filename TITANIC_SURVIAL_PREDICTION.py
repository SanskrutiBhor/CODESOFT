#!/usr/bin/env python
# coding: utf-8

# # CodSoft Internship Task TITANIC_SURVIAL_PREDICTION

# ### Dataset: 
#   * Titanic Dataset

# ### Technologies Used
# * Python
# * Jupyter Notebook
# * Libraries: pandas, numpy, scikit-learn, seaborn, and matplotlib

# ## IMPORT THE LIBRARIES

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2,mutual_info_classif
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# ## READ THE DATASET

# In[30]:


df = pd.read_csv(r"C:\Users\bhors\Desktop\Codesoft\Titanic-Dataset.csv")
df.head()


# ## DATA ANALYSIS

# In[5]:


df.shape


# In[32]:


df.info()


# In[33]:


df.describe()


# In[34]:


df.describe(include="object")


# In[36]:


df.isnull().sum()


# In[37]:


df.duplicated().unique()


# In[38]:


df["Pclass"].unique()


# "Pclass" REPRESENTS TICKET CLASS OF THE PASSENGERS.

# In[39]:


df["Embarked"].unique()


# "Embarked" REPRESENTS ABBREVIATIONS FOR STATIONS.

# In[40]:


data1=df["Survived"].value_counts().reset_index()
data1.columns=["Survived","Count"]
bar1=sns.barplot(x=data1["Survived"],y=data1["Count"],palette=['r','g'])
bar1.bar_label(bar1.containers[0])
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()


# COUNT OF SURVIVORS WAS LESS THAN THE PEOPLE WHO LOST THEIR LIVES

# In[41]:


data2=df["Pclass"].value_counts().reset_index()
data2.columns=["Pclass","Count"]
bar2=sns.barplot(x=data2["Pclass"],y=data2["Count"],palette=['r','g','y'])
bar2.bar_label(bar2.containers[0])
plt.title("Ticket Classes")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.show()


# PEOPLE TRAVELLED MORE IN 3RD CLASS.

# In[42]:


sns.histplot(x=df["Age"],bins=30)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution")
plt.show()


# THIS SHOWS MAXIMUM NUMBER OF PEOPLE TRAVELLED BETWEEN AGE 20-40 YEARS

# In[43]:


data3=df["Sex"].value_counts().reset_index()
data3.columns=["Sex","Count"]
bar3=sns.barplot(x=data3["Sex"],y=data3["Count"],palette=['r','g','y'])
bar3.bar_label(bar3.containers[0])
plt.title("Male vs Female Ratio")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()


# MALES TRAVELLED MORE THAN FEMALES

# In[44]:


data4=df["Embarked"].value_counts().reset_index()
data4.columns=["Embarked","Count"]
bar4=sns.barplot(x=data4["Embarked"],y=data4["Count"],palette=['k','b','m'])
bar4.bar_label(bar4.containers[0])
plt.title("People Going From Various Stations")
plt.xlabel("Stations")
plt.ylabel("Count")
plt.show()


# MAXIMUM NUMBER OF PEOPLE WERE TRAVELLING FROM STATION "S".

# In[45]:


data5=df["SibSp"].value_counts().reset_index()
data5.columns=["SibSp","Count"]
bar5=sns.barplot(x=data5["SibSp"],y=data5["Count"])
bar5.bar_label(bar5.containers[0])
plt.title("Number of Siblings/Spouse Travelling")
plt.xlabel("Siblings/Spouse")
plt.ylabel("Count")
plt.show()


# In[46]:


data6=df["Parch"].value_counts().reset_index()
data6.columns=["Parch","Count"]
bar6=sns.barplot(x=data6["Parch"],y=data6["Count"])
bar6.bar_label(bar6.containers[0])
plt.title("Number of Parents/Children Travelling")
plt.xlabel("Parents/Children")
plt.ylabel("Count")
plt.show()


# BOTH SPOUSES/SIBLINGS AND PARENT/CHILDREN GRAPH SHOWS THAT MAXIMUM PEOPLE TRAVELLED ALONE

# In[47]:


data7=df[["Survived","Embarked"]].value_counts().reset_index()
data7.columns=["Survived","Embarked","Count"]
bar7=sns.barplot(x=data7["Survived"],y=data7["Count"],hue=data7["Embarked"],palette=['m','y','c'])
bar7.bar_label(bar7.containers[0])
bar7.bar_label(bar7.containers[1])
bar7.bar_label(bar7.containers[2])
plt.title("Station Wise Survival")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()


# MAXIMUM PEOPLE WHO DIED WERE FROM STATION "S"

# In[48]:


data8=df[["Survived","Sex"]].value_counts().reset_index()
data8.columns=["Survived","Sex","Count"]
bar8=sns.barplot(x=data8["Survived"],y=data8["Count"],hue=data8["Sex"])
bar8.bar_label(bar8.containers[0])
bar8.bar_label(bar8.containers[1])
plt.title("Male vs Female Survival")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()


# FEMALES SURVIVED MORE THAN MALES

# In[50]:


data9=df[["Survived","Pclass"]].value_counts().reset_index()
data9.columns=["Survived","Pclass","Count"]
bar9=sns.barplot(x=data9["Survived"],y=data9["Count"],hue=data9["Pclass"])
bar9.bar_label(bar9.containers[0])
bar9.bar_label(bar9.containers[1])
bar9.bar_label(bar9.containers[2])
plt.title("Ticket Class Survival")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()


# PEOPLE WHO TRAVELLED IN 3RD CLASS DIED MORE

# In[51]:


sns.displot(x=df["Survived"],y=df["Age"],hue=df["Survived"])
plt.title("Age vs Survival Ratio")
plt.show()


# In[52]:


sns.scatterplot(x=df["SibSp"],y=df["Age"],hue=df["SibSp"])
plt.title("Age vs Spouses/Siblings Ratio")
plt.show()


# PEOPLE BETWEEN 0-30 YEARS TRAVELLED WITH >=4 PEOPLE.

# In[53]:


sns.scatterplot(x=df["Parch"],y=df["Age"],hue=df["Parch"])
plt.title("Age vs Parents/Children Ratio")
plt.show()


# AS YOU CAN SEE PEOPLE AGED BETWEEN 30-70 YEARS TRAVELLED WITH THEIR CHILDEN.

# ## DATA PREPROCESSING

# ENCODING "Sex" AND "Embarked" COLUMN

# In[54]:


df["Sex"]=df["Sex"].map({"male":1,"female":0})
df["Embarked"]=df["Embarked"].map({"S":0,"C":1,"Q":2})


# In[55]:


df.head()


# ENCODING "Name", "Ticket" AND "Cabin" COLUMNS

# In[57]:


le=LabelEncoder()
df["Name"]=le.fit_transform(df["Name"])
df["Ticket"]=le.fit_transform(df["Ticket"])


# In[58]:


df.head()


# HANDLING MISSING VALUES

# In[61]:


df['Age'].fillna(df["Age"].median(), inplace=True)


# In[62]:


df['Age'].isnull().sum()


# In[63]:


sns.boxplot(y=df["Age"])
plt.show()


# In[64]:


df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)


# In[66]:


df["Embarked"].isnull().sum()


# CHANGING DATATYPE TO INTEGER

# In[67]:


df["Age"]=df["Age"].astype(int)
df["Embarked"]=df["Embarked"].astype(int)
df["Fare"]=df["Fare"].astype(int)


# In[68]:


df.info()


# ## FEATURE ENGINEERING

# DROPPING 'Cabin' AS IT HAS TWO MANY NULL VALUES

# In[69]:


df.drop('Cabin',axis=1,inplace=True)
df.head()


# In[70]:


plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),fmt=".2f",annot=True,cmap="Spectral")
plt.title("Correlation Matrix")
plt.show()


# DROPPING "PassengerId", "Name" AND "Ticket" AS THEY ARE JUST UNIQUE IDENTIFIERS

# In[71]:


df.drop(["PassengerId","Name","Ticket"],axis=1,inplace=True)


# In[72]:


features=df.drop("Survived",axis=1)
target=df["Survived"]


# In[73]:


features.head()


# In[74]:


target.head()


# CHI SQUARE TEST

# In[75]:


chi_test=chi2(features,target)


# In[76]:


#F-VALUE
value1=chi_test[0]
f_value=pd.Series(value1)
f_value.index=features.columns
df1=pd.DataFrame({"Feature":f_value.index,"F_Values":f_value})
df1.sort_values(ascending=False,by="F_Values",inplace=True)
df1["F_Values"]=df1["F_Values"].round(3)
bar10=sns.barplot(x="Feature",y="F_Values",data=df1)
bar10.bar_label(bar10.containers[0])
plt.title("F-VALUE")
plt.show()


# In[77]:


# P-VALUE
value2=chi_test[1]
p_value=pd.Series(value2)
p_value.index=features.columns
df2=pd.DataFrame({"Feature":p_value.index,"P_Values":p_value})
df2.sort_values(ascending=True,by="P_Values",inplace=True)
df2["P_Values"]=df2["P_Values"].round(3)
bar11=sns.barplot(x="Feature",y="P_Values",data=df2)
bar11.bar_label(bar11.containers[0])
plt.title("P-VALUE")
plt.show()


# MUTUAL INFORMATION GAIN

# In[78]:


value3=mutual_info_classif(features,target)
mutual_info=pd.Series(value3)
mutual_info.index=features.columns
df3=pd.DataFrame({"Feature":mutual_info.index,"Values":mutual_info})
df3.sort_values(ascending=True,by="Values",inplace=True)
df3["Values"]=df3["Values"].round(3)
bar12=sns.barplot(x="Feature",y="Values",data=df3)
bar12.bar_label(bar12.containers[0])
plt.title("MUTUAL INFORMATION GAIN")
plt.show()


# 'CHI SQUARE TEST' AND 'MUTUAL INFORMATION' CLEARLY DEPICT THE MOST IMPORTANT COLUMNS i.e. 'Sex','Fare','Pclass','Age','Embarked'

# In[79]:


features.drop(["SibSp","Parch"],axis=1,inplace=True)


# In[80]:


features.head()


# ## SPLITTING THE DATA INTO TRAINING AND TESTING DATA

# In[81]:


x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.3,random_state=24)


# In[82]:


x_train.shape


# In[83]:


x_test.shape


# ## MODEL TRAINING

# #### RANDOM FOREST CLASSIFIER

# In[92]:


parameters={"criterion":['gini','entropy'],
            
            "max_depth":[2,6,10,14,18],
            "min_samples_split":[3,7,11,15,19],
            "max_features":["sqrt","log2"],
            "n_estimators":[100,200,300,400]}
model2=RandomForestClassifier()
tuning=GridSearchCV(model2,param_grid=parameters,cv=5,scoring='accuracy')
tuning.fit(x_train,y_train)


# In[93]:


tuning.best_params_


# In[94]:


train_pred2=tuning.predict(x_train)
test_pred2=tuning.predict(x_test)


# ## PERFORMANCE EVALUATION

#  #### ACCURACY

# In[95]:


train_accuracy2=accuracy_score(train_pred2,y_train)
print("Training Accuracy- ",train_accuracy2.round(2))


# In[96]:


test_accuracy2=accuracy_score(test_pred2,y_test)
print("Test Accuracy- ",test_accuracy2.round(2))


# #### CLASSIFICATION REPORT

# In[97]:


report=classification_report(test_pred2,y_test)
print(report)


# #### CONFIDENCE MATRIX

# In[98]:


matrix2=confusion_matrix(test_pred2,y_test)
cm2=ConfusionMatrixDisplay(matrix2,display_labels=["Survived","Not Survived"])
cm2.plot(cmap="crest")
plt.title("Confuson Matrix")
plt.show()


# ## TESTING PREDICTIONS

# In[99]:


import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
pclass=3
sex=0
age=30
fare=9
embarked=0
new_data=[[pclass,sex,age,fare,embarked]]
pred=tuning.predict(new_data)
if(pred[0]==0):
    print("Not Survived")


# In[ ]:




