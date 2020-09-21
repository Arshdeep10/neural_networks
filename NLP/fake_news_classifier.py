import pandas as pd

df = pd.read_csv("NLP/fake-news/train.csv")
df

# get aall indeoendent variables 

x = df.drop('label', axis = 1)
x.head()

#get dependent variable in y
y = df['label']
y.head()

df.shape

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
df = df.dropna()
df.shape

messages = df.copy() # to save the indexing
messages.head()
messages.reset_index(inplace = True)
messages.head()

## text pre processing
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(len(messages)):
    message = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    message = message.lower()
    message = message.split()

    message = [ps.stem(word) for word in message if not word in stopwords.words('english') ]
    meassage = ' '.join(message)
    corpus.append(meassage)


print(corpus[0])


#now applying countvector and apply bag of words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, ngram_range=(1,3))
x = cv.fit_transform(corpus).toarray()
print(cv.get_feature_names()[:20])#check the top 20 feature names 
print(cv.get_params())  #input parameter formed and their types
"""vector formed in bag of words"""
# count_df = pd.DataFrame(x_train, columns = cv.get_feature_names())
# count_df.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()

import numpy as np
from sklearn import metrics
import itertools
y_pred = classifier.fit(x_train, y_train)
score = metrics.accuracy_score(y_test, y_pred)
print("accuracy = ", score)
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

 

