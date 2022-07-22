#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns


# In[3]:


df = pd.read_csv("Amazon_Reviews.csv")
df1 = df[['review__title__text','review__date','review__text']]
df1.columns = ['Title','Date','Review']


# In[4]:


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

#create two new columns
df1['Subjectivity'] = df1['Review'].apply(getSubjectivity)
df1['Polarity'] = df1['Review'].apply(getPolarity)

# Create a function to get the sentiment scores
def getSIA(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

compound=[]
neg=[]
pos=[]
neu=[]
SIA=0

for i in range(0,len(df1['Review'])):
    SIA = getSIA(df1['Review'][i])
    compound.append(SIA['compound'])
    neg.append(SIA['neg'])
    neu.append(SIA['neu'])
    pos.append(SIA['pos'])

# Store the sentiment scores in the merge dataset
df1['Compound']=compound
df1['Negative']=neg
df1['Neutral']=neu
df1['Positive']=pos


# In[5]:


def labelf(comp):
    if comp<=0 :
        label = 0
    elif comp>0:
        label = 1
    return label

lbl = []
for x in df1['Compound']:
    a = labelf(x)
    lbl.append(a)
df1['Label'] = lbl


# In[6]:


X = df1['Review']
y = df1['Label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=24)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

#Vectorizing the text data
ctmTr = cv.fit_transform(X_train)
X_test_dtm = cv.transform(X_test)

from sklearn.linear_model import LogisticRegression
#Training the model
lr = LogisticRegression()
lr.fit(ctmTr, y_train)
#Predicting the labels for test data
y_pred_lr = lr.predict(X_test_dtm)


# In[8]:


#Accuracy score
lr_score = lr.score(X_test_dtm, y_test)
print("Results for Logistic Regression with CountVectorizer")
print(lr_score)


# In[9]:


#Confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr).ravel()

print("\nConfusion matrix")
print(tn, fp, fn, tp)

#True positive and true negative rates
tpr_lr = round(tp/(tp + fn), 4)
tnr_lr = round(tn/(tn+fp), 4)
print("True Positive Rates:",tpr_lr)
print("True Negative Rates:",tpr_lr)


# In[10]:


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cm_lr.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm_lr.flatten()/np.sum(cm_lr)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm_lr, annot=labels, fmt='', cmap='Blues')


# In[ ]:




