import pandas as pd
import sklearn
import matplotlib.pyplot as plt

data = pd.read_csv("creating_artificial_neural_network\Churn_Modelling.csv")
data.head()
data.info()
x = data.iloc[:,3:13]
y = data.iloc[:, 13]

#creating the dummie variables for gender and geography
geography = pd.get_dummies(x["Geography"],drop_first=True)
gender = pd.get_dummies(x["Gender"],drop_first=True)

x = pd.concat([x,geography, gender], axis = 1)
x = x.drop(['Geography', 'Gender'], axis = 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
print("shape",x_train.shape)
print("x_train_info = ")

import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import ReLU, LeakyReLU, ELU
from keras.layers import Dropout

#initiallizing ANN
classifier = Sequential()
classifier.add(Dropout(0.3))

# adding the first layer and first hidden layer 
classifier.add(Dense(units = 6, kernel_initializer= 'he_uniform',activation= 'relu', input_dim = 11 ))
classifier.add(Dropout(0.3))

# Adding the second hidden layer 
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform', activation = 'relu'))
classifier.add(Dropout(0.3))

# Adding the output layer 
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation='sigmoid'))

classifier.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy'])

model_history = classifier.fit(x_train,y_train,validation_split=0.33, batch_size=10, epochs = 100)

print(model_history.history.keys())

# summarize the history
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoches')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()


#summarize the loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoches')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc= 'upper right')
plt.show()

# test the model on test data
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print(score)
