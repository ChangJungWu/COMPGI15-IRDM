
# coding: utf-8

# In[1]:



import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
import time
from gensim.models import word2vec
from gensim import models
import pandas as pd
import numpy as np

start_time = time.time()
    
#pretrained_emb = "c:\input/GoogleNews-vectors-negative300.bin"     
    
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus("c:\input/df_all.txt")
model = word2vec.Word2Vec(sentences,cbow_mean=0,sg=1,window=10,negative=0,hs=1,min_count=3,workers=5,iter=15,size=300)
print('reading data time: {} seconds'.format(round(time.time() - start_time, 2)))  
    
# Save our model.
model.save("c:\input/irdm_model3.bin")

# To load a model.
# model = word2vec.Word2Vec.load("your_mode2.bin")



# In[2]:

y1 = model.similarity("microwav", "cook")
print("microwav和cook的相似度為：", y1)


# In[3]:

print("microwav相似詞前 100 排序")
res = model.most_similar("microwav",topn = 100)
for item in res:
    print(item[0]+","+str(item[1]))


# In[ ]:




# In[ ]:




# In[ ]:



