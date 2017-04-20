
# coding: utf-8

# In[1]:

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


# In[2]:

# Load dataset
start_time = time.time()
df_train = pd.read_csv('c:\input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('c:\input/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('c:\input/product_descriptions.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('c:\input/attributes.csv', encoding="ISO-8859-1")
print('reading data time: {} seconds'.format(round(time.time() - start_time, 2)))
print(df_train.shape[0]+df_test.shape[0])


# In[3]:

# Merge dataset
df_all = pd.concat((df_train, df_test), axis=0, ignore_index='True')
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = df_all.fillna('')
df_attr = df_attr.fillna('')


# # Data Cleaning

# In[4]:

# split special word combination: in.Made -> in made , strongDimensions -> strong Dimensions
def split_special(docs):
    return re.sub(r"(\w)(\.)*([A-Z])", r"\1 \3", str(docs))

# lowercase
def lowercase(docs):
    return str(docs).lower()

## deal with letters
def LetterLetterSplitter(docs):
    docs = re.sub(r"([a-zA-Z]+)[/\-]([a-zA-Z]+)", r"\1 \2", docs)
    return docs

## deal with digits and numbers
def DigitLetterSplitter(docs):
    docs = re.sub(r"(\d+)[\.\-]*([a-zA-Z]+)", r"\1 \2", docs)
    docs = re.sub(r"([a-zA-Z]+)[\.\-]*(\d+)", r"\1 \2", docs)
    return docs
    

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
    docs = docs.replace('&amp;', '&')
    docs = docs.replace('&', '')
    docs = docs.replace('&nbsp;', '')
    docs = docs.replace('&#39;', "'")
    docs = docs.replace('gt/>', '')
    docs = docs.replace('<br', '')
    docs = docs.replace('/>', '')
    docs = docs.replace('/>/Agt/>', '')
    docs = docs.replace('</a<gt/', '')
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
    docs = docs.replace('ten', '10')
    docs = docs.replace('eleven', '11')
    docs = docs.replace('twelve', '12')
    docs = docs.replace('thirteen', '13')
    docs = docs.replace('fourteen', '14')
    docs = docs.replace('fifteen', '15')
    docs = docs.replace('sixteen', '16')
    docs = docs.replace('seventeen', '17')
    docs = docs.replace('eighteen', '18')
    docs = docs.replace('nineteen', '19')
    docs = docs.replace('twenty', '20')
    docs = docs.replace('thirty', '30')
    docs = docs.replace('forty', '40')
    docs = docs.replace('fifty', '50')
    docs = docs.replace('sixty', '60')
    docs = docs.replace('seventy', '70')
    docs = docs.replace('eighty', '80')
    docs = docs.replace('ninety', '90')
    return docs

# normalize units
def normalize_unit(docs):
    docs = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", docs)
    docs = re.sub(r"([0-9]+)( *)(inches|inch|in|in.|')\.?", r"\1 in. ", docs)
    docs = re.sub(r"([0-9]+)( *)(foot|feet|ft|ft.|'')\.?", r"\1 ft. ", docs)
    docs = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb|lb.)\.?", r"\1 lb. ", docs)
    docs = re.sub(r"([0-9]+)( *)(square|sq|sq.) ?\.?(feet|foot|ft|ft.|'')\.?", r"\1 sq.ft. ", docs)
    docs = re.sub(r"([0-9]+)( *)(square|sq|sq.) ?\.?(inches|inch|in|in.|')\.?", r"\ 1 sq.in. ", docs)
    docs = re.sub(r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(feet|foot|ft|ft.|'')\.?", r"\1 cu.ft. ", docs)
    docs = re.sub(r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(inches|inch|in|in.|')\.?", r"\1 cu.in.",docs)
    docs = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1 gal. ", docs)
    docs = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1 oz. ", docs)
    docs = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1 cm. ", docs)
    docs = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1 mm. ", docs)
    docs = re.sub(r"([0-9]+)( *)(minutes|minute)\.?", r"\1 min. " ,docs)
    docs = re.sub(r"([0-9]+)( *)(°|degrees|degree)\.?", r"\1 deg. ",docs)
    docs = re.sub(r"([0-9]+)( *)(v|volts|volt)\.?", r"\1 volt. ", docs)
    docs = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1 watt. ", docs)
    docs = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1 amp. ", docs)
    docs = re.sub(r"([0-9]+)( *)(qquart|quart)\.?", r"\1 qt. ", docs)
    docs = re.sub(r"([0-9]+)( *)(hours|hour|hrs.)\.?", r"\1 hr ", docs)
    docs = re.sub(r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)\.?", r"\1 gal. per min. ", docs)
    docs = re.sub(r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)\.?", r"\1 gal. per hr ", docs)            
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
df_all.iloc[:,[1,4,5]]=df_all.iloc[:,[1,4,5]].applymap(LetterLetterSplitter)
df_all.iloc[:,[1,4,5]]=df_all.iloc[:,[1,4,5]].applymap(DigitLetterSplitter)
df_all.iloc[:,[1,4,5]]=df_all.iloc[:,[1,4,5]].applymap(remove_special)
df_all.iloc[:,[1,4,5]]=df_all.iloc[:,[1,4,5]].applymap(word2num)
df_all.iloc[:,[1,4,5]]=df_all.iloc[:,[1,4,5]].applymap(normalize_unit)
df_all.iloc[:,[1,4,5]]=df_all.iloc[:,[1,4,5]].applymap(stemmer)
df_all.iloc[:,[1,4,5]]=df_all.iloc[:,[1,4,5]].applymap(remove_stopwords)


df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(split_special)
df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(lowercase)
df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(LetterLetterSplitter)
df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(DigitLetterSplitter)
df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(remove_special)
df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(word2num)
df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(normalize_unit)
df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(stemmer)
df_attr.iloc[:,2]=df_attr.iloc[:,2].apply(remove_stopwords)


print('cleaning data time: {} seconds'.format(round(time.time() - start_time, 2)))


# In[7]:

print(df_all.shape)
df_all.to_csv('c:\input/df_all_copy.csv')

print(df_attr.shape)
df_attr.to_csv('c:\input/df_attr.csv')

