#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
df=pd.read_csv("Downloads/adult.csv")


# In[2]:


df


# In[3]:


df['income'].unique()


# In[4]:


df = df.replace({'income': {'<=50K': 0,'>50K': 1}})


# In[5]:


df['income'].unique()


# In[6]:


df


# In[7]:


df['workclass'].mask(df['workclass'] == '?', 'others', inplace=True)
df['occupation'].mask(df['occupation'] == '?', 'others', inplace=True)
df['native.country'].mask(df['native.country'] == '?', 'others', inplace=True)


# In[8]:


df


# In[9]:


df.describe().T


# In[10]:


df.shape


# In[11]:


df.count()


# In[12]:


df.info()


# In[13]:


df.isna().any()


# In[14]:


Age = df['age']
Age.describe().T


# In[15]:


#Feature Segreggation
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']


# In[16]:


numeric_features,categorical_features


# In[17]:


plt.figure(figsize=(40, 50))
plt.suptitle('Univariate Analysis of Categorical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
category = [ 'income', 'workclass','education','sex','relationship','race','marital.status','occupation','native.country']
for i in range(0, len(category)):
    plt.subplot(3, 3, i+1)
    sns.countplot(x=df[category[i]],palette="Set2")
    plt.xlabel(category[i])
    plt.xticks(rotation=45)
    plt.tight_layout() 


# In[18]:


plt.figure(figsize=(40, 50))
plt.suptitle('Univariate Analysis of Numeric Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
category = ['age','education.num','hours.per.week','capital.gain','capital.loss']
for i in range(0, len(category)):
    plt.subplot(3, 3, i+1)
    sns.countplot(x=df[category[i]],palette="Set2")
    plt.xlabel(category[i])
    plt.xticks(rotation=45)
    plt.tight_layout()


# In[19]:


#Bivariate analysis:
sns.heatmap(df.corr(), cmap="Blues", annot=True)


# In[20]:


#Bivariate analysis: Numerical vs Categorical
sns.barplot(x='income', y='age', data=df)


# In[21]:


#sns.catplot(x='income', y='age', kind='swarm', data= df)


# In[22]:


sns.barplot(x='income', y='education',data=df)


# In[23]:


sns.barplot(x='income', y='workclass', data=df)


# In[24]:


sns.barplot(x='income', y='occupation', data=df)


# In[25]:


sns.barplot(x='income', y='race', data=df)


# In[26]:


sns.barplot(x='income', y='education.num', data=df)


# In[27]:


sns.barplot(x='income', y='sex', data=df)


# In[28]:


# % People with income more than 50K
HighIncome = len(df[df['income'] == 1]) / len(df)
print('income:', HighIncome  * 100, '%')


# In[29]:


# % People with income less than equal 50K
LowIncome = len(df[df['income'] == 0]) / len(df)
print('income:', LowIncome  * 100, '%')


# In[30]:


## We can understand from dataset shared that 24% of people are earning income more than 50K & aprox 76% people earn income <=50K


# In[33]:


men = df[df['sex'] == 'Male']
Income_rate_men = len(men[men['income'] == 1]) / len(men)
print('Income_rate_men:', Income_rate_men * 100, '%')


# In[34]:


men = df[df['sex'] == 'Male']
Income_rate_men = len(men[men['income'] == 0]) / len(men)
print('Income_rate_men:', Income_rate_men * 100, '%')


# In[36]:


women = df[df['sex'] == 'Female']
Income_rate_women = len(women[women['income'] == 1]) / len(women)
print('Income_rate_women:', Income_rate_women * 100, '%')


# In[37]:


women = df[df['sex'] == 'Female']
Income_rate_women = len(women[women['income'] == 0]) / len(women)
print('Income_rate_women:', Income_rate_women * 100, '%')


# In[ ]:




