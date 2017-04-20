
# coding: utf-8

# ## Attribute Extraction

# In[1]:

# Import modules
import pandas as pd
import numpy as np
import time
import datetime
import xgboost as xgb
import matplotlib.pyplot as plt
import re, math
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import rcParams
from difflib import SequenceMatcher
from collections import Counter


# In[2]:

# Load dataset
start_time = time.time()
df_train = pd.read_csv('c:\input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('c:\input/test.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('c:\input/df_attr.csv', encoding="ISO-8859-1")
df_all = pd.read_csv('c:\input/df_all_copy.csv', encoding="ISO-8859-1")
print('reading data time: {} seconds'.format(round(time.time() - start_time, 2)))
print(df_all.shape)
df_all.head(5)


# In[3]:


import sys
reload(sys)
sys.setdefaultencoding('utf8')

# Extract attributes index by ranks
start_time = time.time()
attr = df_attr['name'].value_counts()[:].index

# Create columns in df_all.csv about attributes on each product
for attr_index in attr[:11]:
    df_attr_ex = df_attr[df_attr.name == attr_index]
    del df_attr_ex['name']
    df_attr_ex = df_attr_ex.rename(index=str, columns={'product_uid': 'product_uid', 'value': attr_index})
    df_all = pd.merge(df_all, df_attr_ex, how='left', on='product_uid')
    df_all = df_all.fillna('')

# Create a new column 'product_info' storing whole information about each product
df_all['product_info'] = df_all[['product_title','product_description']].astype(str).sum(axis=1)    
    
for attr_index in attr[:11]:
    df_all['space'] = ' '
    df_all['product_info'] = df_all[['product_info', 'space', attr_index]].astype(str).sum(axis=1)
    del df_all['space']

print('producing information about each product: {} seconds'.format(round(time.time() - start_time, 2)))
print(df_all.shape)
df_all.head(5)


# ## Feature Extraction

# In[4]:

# Top 10 attributes in attributes.csv
df_attr['name'].value_counts()[:11]


# In[5]:

# Feature extraction function
def length(docs):
    return len(word_tokenize(docs))

def covered_query_term_num(query, doc):
    query, doc = str(query), str(doc)
    count = 0
    for token in word_tokenize(doc): 
        if token.find(query) >= 0:
            count += 1
    return count

def covered_query_term_ratio(query, doc):
    query, doc = str(query), str(doc)
    count = 0
    for token in word_tokenize(doc): 
        if token.find(query) >= 0:
            count += 1
    return count/len(word_tokenize(doc))

def term_frequency(query, doc):   
    query, doc = str(query), str(doc)
    count = 0
    for token in word_tokenize(doc):
        if query == token:
            count += 1
    return count

def string_similarity(str1, str2):
    str1=str(str1)
    str2=str(str2)
    return SequenceMatcher(None, str1, str2).ratio()

def cosine_similarity(query, doc):
    query, doc = str(query), str(doc)
    WORD = re.compile(r'\w+')
    query, doc = Counter(WORD.findall(query)), Counter(WORD.findall(doc))
    intersection = set(query.keys()) & set(doc.keys())
    numerator = sum([query[x] * doc[x] for x in intersection])
    sum1 = sum([query[x]**2 for x in query.keys()])
    sum2 = sum([doc[x]**2 for x in doc.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


# In[6]:

df_all['query_length'] = df_all['search_term'].astype(str).apply(length)
df_all['product_title_length'] = df_all['product_title'].astype(str).apply(length)
df_all['product_info_length'] = df_all['product_info'].astype(str).apply(length)
df_all['product_description_length'] = df_all['product_description'].astype(str).apply(length)
print('4 features generated')

df_all['covered_query_in_product_descrip_num'] = df_all.apply(lambda x:
                                        covered_query_term_num(x['search_term'], x['product_description']), axis = 1)
df_all['covered_query_in_product_title_num'] = df_all.apply(lambda x:
                                        covered_query_term_num(x['search_term'], x['product_title']), axis = 1)
df_all['covered_query_in_product_info_num'] = df_all.apply(lambda x:
                                        covered_query_term_num(x['search_term'], x['product_info']), axis = 1)
df_all['covered_query_in_product_descrip_ratio'] = df_all.apply(lambda x:
                                        covered_query_term_ratio(x['search_term'], x['product_description']), axis = 1)
df_all['covered_query_in_product_title_ratio'] = df_all.apply(lambda x:
                                        covered_query_term_ratio(x['search_term'], x['product_title']), axis = 1)
df_all['covered_query_in_product_info_ratio'] = df_all.apply(lambda x:
                                        covered_query_term_ratio(x['search_term'], x['product_info']), axis = 1)
print('6 features generated')


# In[7]:

df_all['term_frequency_query_in_product_descrip'] = df_all.apply(lambda x:
                                        term_frequency(x['search_term'], x['product_description']), axis = 1)
df_all['term_frequency_query_in_product_title'] = df_all.apply(lambda x:
                                        term_frequency(x['search_term'], x['product_title']), axis = 1)
df_all['term_frequency_query_in_product_info'] = df_all.apply(lambda x:
                                        term_frequency(x['search_term'], x['product_info']), axis = 1)
print('3 features generated')

df_all['idf_query_in_product_descrip'] = np.log(124428/(1+df_all['term_frequency_query_in_product_descrip']))
df_all['idf_query_in_product_title'] = np.log(124428/(1+df_all['term_frequency_query_in_product_title']))
df_all['idf_query_in_product_info'] = np.log(124428/(1+df_all['term_frequency_query_in_product_info']))
print('3 features generated')

df_all['tf_idf_query_in_product_descrip'] = df_all['term_frequency_query_in_product_descrip']*df_all['idf_query_in_product_descrip']                                   
df_all['tf_idf_query_in_product_title'] = df_all['term_frequency_query_in_product_title']*df_all['idf_query_in_product_title']
df_all['tf_idf_query_in_product_info'] = df_all['term_frequency_query_in_product_info']*df_all['idf_query_in_product_info']
print('3 features generated ')


# In[8]:

df_all['similarity_ratio_query_and_product_descrip'] = df_all.apply(lambda x: string_similarity(x['search_term'], x['product_description']), axis = 1)
df_all['similarity_ratio_query_and_product_title'] = df_all.apply(lambda x: string_similarity(x['search_term'], x['product_title']), axis = 1)
df_all['similarity_ratio_query_and_product_info'] = df_all.apply(lambda x: string_similarity(x['search_term'], x['product_info']), axis = 1)
print('3 features generated')


# In[9]:

df_all['cosine_similarity_query_and_product_descrip'] = df_all.apply(lambda x: cosine_similarity(x['search_term'], x['product_description']), axis = 1)
df_all['cosine_similarity_query_and_product_title'] = df_all.apply(lambda x: cosine_similarity(x['search_term'], x['product_title']), axis = 1)
df_all['cosine_similarity_query_and_product_info'] = df_all.apply(lambda x: cosine_similarity(x['search_term'], x['product_info']), axis = 1)
print('3 features generated')


# In[10]:

print(df_all.shape)
#df_all = df_all.rename(index=str, columns={'similarity_rati_query_and_product_info': 'similarity_ratio_query_and_product_info'})
df_all.head(20)


# In[11]:

df_all.to_csv('c:\input/df_all.csv')


# In[36]:

df_all = pd.read_csv('c:\input/df_all.csv', low_memory=False)
df_all.shape


# In[37]:

def normalisation(df):
    df_norm = (df - df.min()) / (df.max() - df.min())
    return df_norm


# In[38]:

def create_result(prediction, info):
    result = pd.DataFrame()
    result['id'] = df_all['id'][num_train:].astype(str)
    prediction[prediction > 3.0] = 3.0
    prediction[prediction < 1.0 ] = 1.0
    result['relevance'] = prediction
    result.to_csv(info, index = False)


# In[39]:

num_train = df_train.shape[0]
train_x = df_all.iloc[:num_train,31:]
test_x  = df_all.iloc[num_train:,31:]


# In[40]:

train_x.dtypes


# In[41]:

# Normalisation features before learning
train_x = normalisation(train_x)
train_y = df_all['relevance'][:num_train].values
test_x = normalisation(test_x).values


# In[50]:

print(train_x.shape)
print(train_y.shape)
pd.DataFrame(train_x).head(5)


# In[51]:

print(test_x.shape)
pd.DataFrame(test_x).head(5)


# ## Model

# In[44]:

#Lasso regularized linear model
#Set up alpha for regularization
alphas = np.linspace(0.01, 100, num=10000)
alphas = alphas.tolist()
model_lasso = LassoCV(alphas = alphas).fit(train_x, train_y)
test_y_lasso = model_lasso.predict(test_x)
create_result(test_y_lasso, 'lasso_regress.csv')

#Ridge regularized linear model
model_ridge = RidgeCV(alphas = alphas).fit(train_x, train_y)
test_y_ridge = model_ridge.predict(test_x)
create_result(test_y_ridge, 'ridge_regress.csv')

# Xgboost model
model_xgb = xgb.XGBRegressor(n_estimators=500, max_depth=10, learning_rate=0.1).fit(train_x, train_y)
test_y_xgb = model_xgb.predict(test_x)
create_result(test_y_ridge, 'xgboost.csv')

# Random forest model
model_rf = RandomForestRegressor(n_estimators=10).fit(train_x, train_y)
test_y_rf = model_rf.predict(test_x)
create_result(test_y_rf, 'rand_forest.csv')

# Gradient boost model
model_grad_boost = GradientBoostingRegressor(random_state=0, loss='ls').fit(train_x, train_y)
test_y_grad_boost = model_grad_boost.predict(test_x)
create_result(test_y_grad_boost, 'grad_boost.csv')

# KNN model
model_knn = KNeighborsRegressor().fit(train_x, train_y)
test_y_knn = model_knn.predict(test_x)
create_result(test_y_knn, 'knn.csv')

# SVM model
model_svm = SVR(C=1, epsilon=0.2).fit(train_x, train_y)
test_y_svm = model_svm.predict(test_x)
create_result(test_y_svm, 'svm.csv')


# In[ ]:

get_ipython().magic(u'pylab inline')
rcParams['figure.figsize'] = (12.0, 6.0)
data=[test_y_lasso, test_y_ridge, test_y_rf,test_y_knn, test_y_grad_boost, test_y_xgb, test_y_svm]
plt.figure()
plt.boxplot(data)
plt.xticks([1,2,3,4,5,6,7],('lasso', 'ridge', 'random forest', 'knn', 'gradient boost', 'xgboost', 'svm'))

