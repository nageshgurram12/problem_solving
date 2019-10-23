
# coding: utf-8

# ### What's cooking (kaggle challenge)

# In[64]:


import pandas as pd
import numpy as np

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# #### Read the JSON training and test data :

# In[35]:


training_data = pd.read_json("train.json")
testing_data = pd.read_json("test.json")


# #### Preprocess the data:
# 
# convert all the ingredients to lower case and remove any special chars and then join into a single string

# In[38]:


stemmer = WordNetLemmatizer()
def preprocess(data):
    all_ingredients = data["ingredients"]
    all_ingre_strs = []
    for ingredients in all_ingredients:
        ingre_str = ''
        for i in ingredients:
            
            # convert every word to lower case
            i = str.lower(i)

            # remove non alpha bet
            i = re.sub('[^A-Za-z]',' ', i)
            
            ingre_str = ingre_str + ' ' + i
        all_ingre_strs.append(ingre_str)
    #print(all_ingre_strs)
    data["ingre_str"] = all_ingre_strs

preprocess(training_data)
preprocess(testing_data)


# #### Create a validation set from training data:

# In[42]:


training_data, val_data = train_test_split(training_data, test_size=0.1)


# #### Transform ingredients into word count frequency matrix:

# In[47]:


vectorizer = CountVectorizer(stop_words='english', analyzer="word")
training_feat = vectorizer.fit_transform(training_data["ingre_str"])
testing_feat = vectorizer.transform(testing_data["ingre_str"])
val_feat = vectorizer.transform(val_data["ingre_str"])


# #### Use decision tree classifier to predict the cuisine for test samples:
# 
# Set pre-pruning condition to stop the splitting the node when samples reached size below 10

# In[73]:


dt_classifier = DecisionTreeClassifier(min_samples_split=10)
dt_classifier.fit(training_feat, training_data["cuisine"])


# In[74]:


val_predict = dt_classifier.predict(val_feat)
accuracy_score(val_data["cuisine"], val_predict)


# In[62]:


testing_data["cuisine"] = dt_classifier.predict(testing_feat)
testing_data[["id", "cuisine"]].to_csv("kaggle-DT.csv", index=False)


# #### Use KNN classifier to predict the cuisine:

# In[76]:


knn_classifier = KNeighborsClassifier(n_neighbors=10)
knn_classifier.fit(training_feat, training_data["cuisine"])


# In[77]:


val_predict = knn_classifier.predict(val_feat)
accuracy_score(val_data["cuisine"], val_predict)


# In[78]:


testing_data["cuisine"] = knn_classifier.predict(testing_feat)
testing_data[["id", "cuisine"]].to_csv("kaggle-KNN.csv", index=False)

