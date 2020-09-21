from tensorflow.keras.preprocessing.text import one_hot
### sentences
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]

print(sent)

voc_size = 1000

#one hot representation 
onehot_representation = [one_hot(words, voc_size)for words in sent]
print(onehot_representation)


#word embedding 
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np 

set_len = 8
embedding_docs = pad_sequences(onehot_representation,padding='pre', maxlen=set_len)
print(embedding_docs)


dim = 5
model = Sequential()
model.add(Embedding(voc_size, dim, input_length=set_len))
model.compile('adam', 'mse')
model.summary()

print(model.predict(embedding_docs))
