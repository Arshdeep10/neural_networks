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


## model compile using hyper parameters
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation, ReLU, Embedding, ELU, Flatten, LeakyReLU, BatchNormalization
from keras.activations import relu, sigmoid
from keras.layers import Dropout


def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, input_dim = x_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))

        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))


    model.add(Dense(units = 1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = KerasClassifier(build_fn = create_model, verbose = 0)

layers = ([20], [20,15,15], [45,30,15])
activation = ('sigmoid', 'relu')
pram_grid = dict( layers = layers, activation = activation, batch_size = (120,256), epochs = [30])
grid = GridSearchCV(estimator = model, param_grid = pram_grid, cv = 5) ## cv = cross validation 
grid_result = grid.fit(x_train, y_train)


print(grid.best_score_, grid.best_params_)
grid.predict(x_test)
y_pred = (y_pred > 0.5)


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)


