
# coding: utf-8

# ## Attribute Extraction

# In[ ]:

# Import modules
import pandas as pd
import numpy as np
import time
import datetime
import xgboost as xgb
import matplotlib.pyplot as plt
import re, math
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import rcParams
from difflib import SequenceMatcher
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
import logging
import time
from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec
from gensim import models
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


# In[ ]:

# Load dataset
start_time = time.time()
df_train = pd.read_csv('input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('input/test.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('df_attr.csv', encoding="ISO-8859-1")
df_all = pd.read_csv('df_all_clean.csv', encoding="ISO-8859-1")
print('reading data time: {} seconds'.format(round(time.time() - start_time, 2)))
print(df_all.shape)
df_all.head(5)


# In[ ]:

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
df_all.columns


# In[ ]:

del df_all['Unnamed: 0_x']
del df_all['Unnamed: 0_y']
del df_all['Unnamed: 0']
df_all.columns


# ## Feature Extraction

# In[ ]:

# Top 10 attributes in attributes.csv
df_attr['name'].value_counts()[:11]


# In[ ]:

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

def word_2_vec(query):
    start_time = time.time()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    query = str(query)
    sentences = word2vec.Text8Corpus("df_all.csv")
    model = word2vec.Word2Vec(sentences, size=250)
    model.build_vocab(sentences)
    for token in word_tokenize(each_query):
        return model.most_similar(token, topn = 2)[0][1]


# In[ ]:

df_all['query_length'] = df_all['search_term'].astype(str).apply(length)
df_all['product_title_length'] = df_all['product_title'].astype(str).apply(length)
df_all['product_info_length'] = df_all['product_info'].astype(str).apply(length)
df_all['product_description_length'] = df_all['product_description'].astype(str).apply(length)
print('4 features generated')


# In[ ]:

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


# In[ ]:

df_all['word_2_vec'] = df_all['search_term'].apply(word_2_vec)


# In[ ]:

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


# In[ ]:

df_all['similarity_ratio_query_and_product_descrip'] = df_all.apply(lambda x: string_similarity(x['search_term'], x['product_description']), axis = 1)
df_all['similarity_ratio_query_and_product_title'] = df_all.apply(lambda x: string_similarity(x['search_term'], x['product_title']), axis = 1)
df_all['similarity_ratio_query_and_product_info'] = df_all.apply(lambda x: string_similarity(x['search_term'], x['product_info']), axis = 1)
print('3 features generated')


# In[ ]:

df_all['cosine_similarity_query_and_product_descrip'] = df_all.apply(lambda x: cosine_similarity(x['search_term'], x['product_description']), axis = 1)
df_all['cosine_similarity_query_and_product_title'] = df_all.apply(lambda x: cosine_similarity(x['search_term'], x['product_title']), axis = 1)
df_all['cosine_similarity_query_and_product_info'] = df_all.apply(lambda x: cosine_similarity(x['search_term'], x['product_info']), axis = 1)
print('3 features generated')


# In[ ]:

print(df_all.shape)
df_all.iloc[:,18:].skew()


# In[ ]:

df_all.to_csv('df_all.csv')


# In[ ]:

df_all = pd.read_csv('df_all.csv', low_memory=False)
print(df_all.shape)
df_all.columns


# In[ ]:

df_all.iloc[:,18:].columns


# In[ ]:

def normalisation(df):
    df_norm = (df - df.min()) / (df.max() - df.min())
    return df_norm


# In[ ]:

def predict_to_result(prediction, info):
    result = pd.DataFrame()
    result['id'] = df_all['id'][num_train:].astype(str)
    prediction[prediction > 3.0] = 3.0
    prediction[prediction < 1.0 ] = 1.0
    result['relevance'] = prediction
    result.to_csv(info, index = False)


# In[ ]:

num_train = df_train.shape[0]
train_x = df_all.iloc[:num_train,18:]
test_x  = df_all.iloc[num_train:,18:]


# In[ ]:

# Normalisation features before learning
train_x = normalisation(train_x).values
train_y = df_all['relevance'][:num_train].values
test_x = normalisation(test_x).values


# In[ ]:

print(train_x.shape)
print(train_y.shape)
pd.DataFrame(train_x)


# In[ ]:

print(test_x.shape)
pd.DataFrame(test_x)


# ## Model

# In[ ]:

#Lasso regularized linear model
alphas = np.linspace(0.01, 100, num=10000)
alphas = alphas.tolist()
model_lasso = LassoCV(alphas = alphas).fit(train_x, train_y)
test_y_lasso = model_lasso.predict(test_x)
predict_to_result(test_y_lasso, 'lasso_regress.csv')

df_solution = pd.read_csv('solution.csv')
df_lasso = pd.read_csv('lasso_regress.csv')
df_solution = pd.merge(df_solution, df_lasso, how='left', on='id')
df_solution = df_solution[df_solution['Usage'] =='Public']

mae_la = mean_absolute_error(df_solution.iloc[:,1], df_solution.iloc[:,3])
rmse_la = math.sqrt(mean_squared_error(df_solution.iloc[:,1], df_solution.iloc[:,3]))
rmse_la, mae_la


# In[ ]:

#Ridge regularized linear model
alphas = np.linspace(0.01, 100, num=10000)
alphas = alphas.tolist()
model_ridge = RidgeCV(alphas = alphas).fit(train_x, train_y)
test_y_ridge = model_ridge.predict(test_x)
predict_to_result(test_y_ridge, 'ridge_regress.csv')

df_solution = pd.read_csv('solution.csv')
df_ridge = pd.read_csv('ridge_regress.csv')
df_solution = pd.merge(df_solution, df_ridge, how='left', on='id')
df_solution = df_solution[df_solution['Usage'] =='Public']

mae_rg = mean_absolute_error(df_solution.iloc[:,1], df_solution.iloc[:,3])
rmse_rg = math.sqrt(mean_squared_error(df_solution.iloc[:,1], df_solution.iloc[:,3]))
rmse_rg, mae_rg


# In[ ]:

# Xgboost model
model_xgb = xgb.XGBRegressor(n_estimators=29, max_depth=4).fit(train_x, train_y)
test_y_xgb = model_xgb.predict(test_x)
predict_to_result(test_y_xgb, 'xgboost.csv')

df_solution = pd.read_csv('solution.csv')
df_xgb = pd.read_csv('xgboost.csv')
df_solution = pd.merge(df_solution, df_xgb, how='left', on='id')
df_solution = df_solution[df_solution['Usage'] =='Public']

mae_xg = mean_absolute_error(df_solution.iloc[:,1], df_solution.iloc[:,3])
rmse_xg = math.sqrt(mean_squared_error(df_solution.iloc[:,1], df_solution.iloc[:,3]))
rmse_xg, mae_xg


# In[ ]:

# Random forest model
model_rf = RandomForestRegressor(n_estimators=10).fit(train_x, train_y)
test_y_rf = model_rf.predict(test_x)
predict_to_result(test_y_rf, 'rand_forest.csv')

df_solution = pd.read_csv('solution.csv')
df_rf = pd.read_csv('rand_forest.csv')
df_solution = pd.merge(df_solution, df_rf, how='left', on='id')
df_solution = df_solution[df_solution['Usage'] =='Public']

mae_rf = mean_absolute_error(df_solution.iloc[:,1], df_solution.iloc[:,3])
rmse_rf = math.sqrt(mean_squared_error(df_solution.iloc[:,1], df_solution.iloc[:,3]))
rmse_rf, mae_rf


# In[ ]:

# Gradient boost model
model_grad_boost = GradientBoostingRegressor(random_state=0, loss='ls').fit(train_x, train_y)
test_y_grad_boost = model_grad_boost.predict(test_x)
predict_to_result(test_y_grad_boost, 'grad_boost.csv')

df_solution = pd.read_csv('solution.csv')
df_gb = pd.read_csv('grad_boost.csv')
df_solution = pd.merge(df_solution, df_gb, how='left', on='id')
df_solution = df_solution[df_solution['Usage'] =='Public']

mae_gb = mean_absolute_error(df_solution.iloc[:,1], df_solution.iloc[:,3])
rmse_gb = math.sqrt(mean_squared_error(df_solution.iloc[:,1], df_solution.iloc[:,3]))
rmse_gb, mae_gb


# In[ ]:

# KNN model
model_knn = KNeighborsRegressor().fit(train_x, train_y)
test_y_knn = model_knn.predict(test_x)
predict_to_result(test_y_knn, 'knn.csv')

df_solution = pd.read_csv('solution.csv')
df_knn = pd.read_csv('knn.csv')
df_solution = pd.merge(df_solution, df_knn, how='left', on='id')
df_solution = df_solution[df_solution['Usage'] =='Public']

mae_knn = mean_absolute_error(df_solution.iloc[:,1], df_solution.iloc[:,3])
rmse_knn = math.sqrt(mean_squared_error(df_solution.iloc[:,1], df_solution.iloc[:,3]))
rmse_knn, mae_knn


# In[ ]:

# SVM model
model_svm = SVR(C=1, epsilon=0.2).fit(train_x, train_y)
test_y_svm = model_svm.predict(test_x)
predict_to_result(test_y_svm, 'svm.csv')

df_solution = pd.read_csv('solution.csv')
df_svm = pd.read_csv('svm.csv')
df_solution = pd.merge(df_solution, df_svm, how='left', on='id')
df_solution = df_solution[df_solution['Usage'] =='Public']

mae_sv = mean_absolute_error(df_solution.iloc[:,1], df_solution.iloc[:,3])
rmse_sv = math.sqrt(mean_squared_error(df_solution.iloc[:,1], df_solution.iloc[:,3]))
rmse_sv, mae_sv


# In[ ]:

get_ipython().magic('pylab inline')
rcParams['figure.figsize'] = (12.0, 6.0)
data=[test_y_lasso, test_y_ridge, test_y_rf,test_y_knn, test_y_grad_boost, test_y_xgb, test_y_svm]
plt.figure()
plt.boxplot(data)
plt.xticks([1,2,3,4,5,6,7],('lasso', 'ridge', 'random forest', 'knn', 'gradient boost', 'xgboost', 'svm'))


# ## Evaluation

# In[ ]:

peter= pd.read_csv('Peter.csv')
del peter['Unnamed: 0']
del peter['product_uid']
peter.colums


# In[ ]:

train_x = peter[:num_train]
test_x = peter[num_train:]
train_x.shape, test_x.shape


# In[ ]:

#Lasso regularized linear model
alphas = np.linspace(0.01, 100, num=10000)
alphas = alphas.tolist()
model_lasso = LassoCV(alphas = alphas).fit(train_x, train_y)
test_y_lasso = model_lasso.predict(test_x)
predict_to_result(test_y_lasso, 'lasso_regress.csv')

df_solution = pd.read_csv('solution.csv')
df_lasso = pd.read_csv('lasso_regress.csv')
df_solution = pd.merge(df_solution, df_lasso, how='left', on='id')
df_solution = df_solution[df_solution['Usage'] =='Public']

mae_la = mean_absolute_error(df_solution.iloc[:,1], df_solution.iloc[:,3])
rmse_la = math.sqrt(mean_squared_error(df_solution.iloc[:,1], df_solution.iloc[:,3]))
rmse_la, mae_la


# In[ ]:

#Ridge regularized linear model
alphas = np.linspace(0.01, 100, num=10000)
alphas = alphas.tolist()
model_ridge = RidgeCV(alphas = alphas).fit(train_x, train_y)
test_y_ridge = model_ridge.predict(test_x)
predict_to_result(test_y_ridge, 'ridge_regress.csv')

df_solution = pd.read_csv('solution.csv')
df_ridge = pd.read_csv('ridge_regress.csv')
df_solution = pd.merge(df_solution, df_ridge, how='left', on='id')
df_solution = df_solution[df_solution['Usage'] =='Public']

mae_rg = mean_absolute_error(df_solution.iloc[:,1], df_solution.iloc[:,3])
rmse_rg = math.sqrt(mean_squared_error(df_solution.iloc[:,1], df_solution.iloc[:,3]))
rmse_rg, mae_rg


# In[ ]:

# Xgboost model
model_xgb = xgb.XGBRegressor().fit(train_x, train_y)
test_y_xgb = model_xgb.predict(test_x)
predict_to_result(test_y_xgb, 'xgboost.csv')

df_solution = pd.read_csv('solution.csv')
df_xgb = pd.read_csv('xgboost.csv')
df_solution = pd.merge(df_solution, df_xgb, how='left', on='id')
df_solution = df_solution[df_solution['Usage'] =='Public']

mae_xg = mean_absolute_error(df_solution.iloc[:,1], df_solution.iloc[:,3])
rmse_xg = math.sqrt(mean_squared_error(df_solution.iloc[:,1], df_solution.iloc[:,3]))
rmse_xg, mae_xg


# In[ ]:

# Random forest model
model_rf = RandomForestRegressor(n_estimators=10).fit(train_x, train_y)
test_y_rf = model_rf.predict(test_x)
predict_to_result(test_y_rf, 'rand_forest.csv')

df_solution = pd.read_csv('solution.csv')
df_rf = pd.read_csv('rand_forest.csv')
df_solution = pd.merge(df_solution, df_rf, how='left', on='id')
df_solution = df_solution[df_solution['Usage'] =='Public']

mae_rf = mean_absolute_error(df_solution.iloc[:,1], df_solution.iloc[:,3])
rmse_rf = math.sqrt(mean_squared_error(df_solution.iloc[:,1], df_solution.iloc[:,3]))
rmse_rf, mae_rf


# In[ ]:

# Gradient boost model
model_grad_boost = GradientBoostingRegressor(random_state=0, loss='ls').fit(train_x, train_y)
test_y_grad_boost = model_grad_boost.predict(test_x)
predict_to_result(test_y_grad_boost, 'grad_boost.csv')

df_solution = pd.read_csv('solution.csv')
df_gb = pd.read_csv('grad_boost.csv')
df_solution = pd.merge(df_solution, df_gb, how='left', on='id')
df_solution = df_solution[df_solution['Usage'] =='Public']

mae_gb = mean_absolute_error(df_solution.iloc[:,1], df_solution.iloc[:,3])
rmse_gb = math.sqrt(mean_squared_error(df_solution.iloc[:,1], df_solution.iloc[:,3]))
rmse_gb, mae_gb


# In[ ]:

# KNN model
model_knn = KNeighborsRegressor().fit(train_x, train_y)
test_y_knn = model_knn.predict(test_x)
predict_to_result(test_y_knn, 'knn.csv')

df_solution = pd.read_csv('solution.csv')
df_knn = pd.read_csv('knn.csv')
df_solution = pd.merge(df_solution, df_knn, how='left', on='id')
df_solution = df_solution[df_solution['Usage'] =='Public']

mae_knn = mean_absolute_error(df_solution.iloc[:,1], df_solution.iloc[:,3])
rmse_knn = math.sqrt(mean_squared_error(df_solution.iloc[:,1], df_solution.iloc[:,3]))
rmse_knn, mae_knn


# In[ ]:

# SVM model
model_svm = SVR(C=1, epsilon=0.2).fit(train_x, train_y)
test_y_svm = model_svm.predict(test_x)
predict_to_result(test_y_svm, 'svm.csv')

df_solution = pd.read_csv('solution.csv')
df_svm = pd.read_csv('svm.csv')
df_solution = pd.merge(df_solution, df_svm, how='left', on='id')
df_solution = df_solution[df_solution['Usage'] =='Public']

mae_sv = mean_absolute_error(df_solution.iloc[:,1], df_solution.iloc[:,3])
rmse_sv = math.sqrt(mean_squared_error(df_solution.iloc[:,1], df_solution.iloc[:,3]))
rmse_sv, mae_sv


# ## Evaluation

# In[ ]:

get_ipython().magic('pylab inline')
rcParams['figure.figsize'] = (12.0, 6.0)
data=[test_y_lasso, test_y_ridge, test_y_rf,test_y_knn, test_y_grad_boost, test_y_xgb, test_y_svm]
plt.figure()
plt.boxplot(data)
plt.xticks([1,2,3,4,5,6,7],('lasso', 'ridge', 'random forest', 'knn', 'gradient boost', 'xgboost', 'svm'))


# In[ ]:

# data to plot
n_groups = 7
means_frank = (rmse_la, rmse_rg, rmse_rf, rmse_knn, rmse_gb, rmse_xg, rmse_sv)
means_guido = (mae_la, mae_rg, mae_rf, mae_knn, mae_gb, mae_xg, mae_sv)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.4
opacity = 0.7
 
rects1 = plt.bar(index, means_frank, bar_width,
                 alpha=opacity,
                 color='b',
                 label='rmse')
 
rects2 = plt.bar(index + bar_width, means_guido, bar_width,
                 alpha=opacity,
                 color='g',
                 label='mae')
 
#plt.xlabel('Scores')
plt.ylabel('Error')
plt.title('Scores by model')
plt.xticks(index + bar_width, ('lasso', 'ridge', 'random forest', 'knn', 'gradient boost', 'xgboost', 'svm'))
plt.legend()
 
plt.tight_layout()
plt.show()
plt.show()

