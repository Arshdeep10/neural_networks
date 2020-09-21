import pandas as pd
import re

messages = pd.read_csv('NLP\smsspamcollection\SMSSpamCollection', sep = '\t', names= ['lable', 'message'])
print(messages)

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  #can use lamitization also
ps = PorterStemmer()

corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer  # can use tfidf and bag of words also 
cv = CountVectorizer(max_features=2500)
x = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['lable'])
y = y.iloc[:,1].values# to remove 1 column


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)


#Train the model using naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detection_model = MultinomialNB().fit(x_train,y_train)

y_pred = spam_detection_model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)
print(confusion)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
