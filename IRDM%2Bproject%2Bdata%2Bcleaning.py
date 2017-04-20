
# coding: utf-8

# In[ ]:

# Import modules
import pandas as pd
import numpy as np
import time
import datetime
import re
from google_dict import *
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[ ]:

# Load dataset
start_time = time.time()
df_train = pd.read_csv('input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('input/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('input/product_descriptions.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('input/attributes.csv', encoding="ISO-8859-1")
print('reading data time: {} seconds'.format(round(time.time() - start_time, 2)))
print(df_train.shape[0]+df_test.shape[0])


# In[ ]:

# Merge dataset
df_all = pd.concat((df_train, df_test), axis=0, ignore_index='True')
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = df_all.fillna('')
df_attr = df_attr.fillna('')


# # Data Cleaning

# In[ ]:

# split special word combination: in.Made -> in made , strongDimensions -> strong Dimensions
def split_special(docs):
    return re.sub(r"(\w)(\.)*([A-Z])", r"\1 \3", str(docs))

# lowercase
def lowercase(docs):
    return str(docs).lower()

# remove special symbal
def remove_special(docs):
    docs = docs.replace('#', '')
    docs = docs.replace('!', '')
    docs = docs.replace('"', '')
    docs = docs.replace('$', '')
    docs = docs.replace('&', '')
    docs = docs.replace('\'', '')
    docs = docs.replace('(', '')
    docs = docs.replace(')', '')
    docs = docs.replace('*', '')
    docs = docs.replace('+', '')
    docs = docs.replace(':', '')
    docs = docs.replace(';', '')
    docs = docs.replace('<', '')
    docs = docs.replace('=', '')
    docs = docs.replace('>', '')
    docs = docs.replace('?', '')
    docs = docs.replace('@', '')
    docs = docs.replace('[', '')
    docs = docs.replace(']', '')
    docs = docs.replace('^', '')
    docs = docs.replace('_', '')
    docs = docs.replace('{', '')
    docs = docs.replace('|', '')
    docs = docs.replace('}', '')
    docs = docs.replace('~', '')
    docs = docs.replace('`', '')
    docs = docs.replace('â', '')
    docs = docs.replace('&amp', '')
    docs = docs.replace('&nbsp', '')
    docs = docs.replace('&#39', '')
    docs = docs.replace('gt/>', '')
    docs = docs.replace('<br', '')
    return docs

# word to number
def word2num(docs):
    docs = docs.replace('zero', '0')
    docs = docs.replace('one', '1')
    docs = docs.replace('two', '2')
    docs = docs.replace('three', '3')
    docs = docs.replace('four', '4')
    docs = docs.replace('five', '5')
    docs = docs.replace('six', '6')
    docs = docs.replace('seven', '7')
    docs = docs.replace('eight', '8')
    docs = docs.replace('nine', '9')
    return docs

# normalize units
def normalize_unit(docs):
    docs = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", docs)
    docs = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1inch. ", docs)
    docs = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", docs)
    docs = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", docs)
    docs = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq ft ", docs)
    docs = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu ft ", docs)
    docs = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", docs)
    docs = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", docs)
    docs = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", docs)
    docs = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", docs)
    docs = docs.replace("°"," degrees ")
    docs = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", docs)
    docs = docs.replace(" v "," volts ")
    docs = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", docs)
    docs = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", docs)
    docs = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", docs)
    return docs

# Stem: cats -> cat, interesting -> interest, happily -> happi
def stemmer(docs):
    stemmer = SnowballStemmer('english', ignore_stopwords='True').stem
    return ' '.join([stemmer(word)
                    for word in word_tokenize(docs)]) 

# remove stopwords
def remove_stopwords(docs):
    stopset = stopwords.words('english')
    return ' '.join([word
                    for word in word_tokenize(docs) if word not in stopset])

start_time = time.time()
# query spelling correction: using the Google dict
# from the forum
# https://www.kaggle.com/steubk/home-depot-product-search-relevance/fixing-typos
df_all['search_term']=df_all['search_term'].map(lambda x: google_dict[x] if x in google_dict.keys() else x)  

df_all.iloc[:,[1,4,5]]=df_all.iloc[:,[1,4,5]].applymap(split_special)
df_all.iloc[:,[1,4,5]]=df_all.iloc[:,[1,4,5]].applymap(lowercase)
df_all.iloc[:,[1,4,5]]=df_all.iloc[:,[1,4,5]].applymap(remove_special)
df_all.iloc[:,[1,4,5]]=df_all.iloc[:,[1,4,5]].applymap(word2num)
df_all.iloc[:,[1,4,5]]=df_all.iloc[:,[1,4,5]].applymap(normalize_unit)
df_all.iloc[:,[1,4,5]]=df_all.iloc[:,[1,4,5]].applymap(stemmer)
df_all.iloc[:,[1,4,5]]=df_all.iloc[:,[1,4,5]].applymap(remove_stopwords)

df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(split_special)
df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(lowercase)
df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(remove_special)
df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(word2num)
df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(normalize_unit)
df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(stemmer)
df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(remove_stopwords)

print('cleaning data time: {} seconds'.format(round(time.time() - start_time, 2)))


# In[ ]:

print(df_all.shape)
df_all.to_csv('df_all_copy.csv')

