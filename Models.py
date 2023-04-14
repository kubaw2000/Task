# import necessary packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
import matplotlib.pyplot as plt
import joblib

###   1. Load the Covertype Dataset
data=pd.read_csv('covtype.data')

# seperate labels
X=data.iloc[:,:-1]
y=data.iloc[:,-1]

# split into training set, validation set and test set

X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.4, random_state=21)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=21)

# normalize
mean=X_train.mean()
std=X_train.std()
X_train=(X_train-mean)/std
X_val=(X_val-mean)/std
X_test=(X_test-mean)/std

#normalisation is needed for heuristic, knn and neural network, not necessary for decision tree

###   2. Implement a very simple heuristic that will classify the data

##  We will classify data by minimal distance of normalized data to the means of classes


class Heuristic:
    def __init__(self):
        pass
    def fit(self, X_train, y_train):
        # get means from training set
        self.labels = np.sort(y_train.unique())
        self.means = [X_train[y_train == y].mean() for y in self.labels]
    def predict(self,X_test):
        return X_test.apply(lambda row: np.argmin([np.linalg.norm(row- self.means[y - 1]) for y in self.labels]) + 1, axis=1)

# build heuristic model
heuristic=Heuristic()
heuristic.fit(X_train, y_train)

# save model
joblib.dump(heuristic, 'heuristic.pkl')

#classify data
y_pred=heuristic.predict(X_val)
score=accuracy_score(y_val, y_pred)
print('Accuracy for heuristic model on validation set:  {}'.format(score))

### 3. Use Scikit-learn library to train two simple Machine Learning models

## K_nearest neighbors. Long to train due to large size of a dataset, but highly accurate

# build model and train
KNN=KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train, y_train)

#save model
joblib.dump(KNN, 'knn.pkl')

# classify and get the accuracy
y_pred=KNN.predict(X_val)
score=accuracy_score(y_val, y_pred)
print('Accuracy for KNN model on validation set:  {}'.format(score))

## Decision Tree. Fast to train combined with high accuracy

# build model and train
Tree=DecisionTreeClassifier(random_state=21)
Tree.fit(X_train, y_train)

#save model
joblib.dump(Tree, 'tree.pkl')

# classify and get the accuracy
y_pred=Tree.predict(X_val)
score=accuracy_score(y_val, y_pred)
print('Accuracy for Decision Tree model on validation set:  {}'.format(score))


### 4. Use TensorFlow library to train a neural network that will classify the data

##  Find hyperparameters

# one-hot encode labels
y_train=keras.utils.to_categorical(y_train)
y_val=keras.utils.to_categorical(y_val)



# create a set of hyperparameters to choose from
parameters={'architecture': [(256,256,256,256), (128,64,256,128), (64,128,128,128,64), (128,256,256,128), (64,128,256,256,128,64),
                             (64,256,128,128,256,64), (128,128,128,128,128)],
            'dropout': [0,0.1], 'lr': [0.001,0.005,0.01], 'batch': [32,64,128]}

# define a function which will build model given hyperparameters
def create_model(arch,drop_rate,lr):
  model=keras.models.Sequential([keras.layers.Input(shape=(X_train.shape[1]))])
  for l in range(len(arch)):
    model.add(keras.layers.Dense(units=arch[l], activation='relu'))
    model.add(keras.layers.Dropout(drop_rate))
  model.add(keras.layers.Dense(units=y_train.shape[1], activation='softmax'))
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

#define a function which will find the best hyperparameters
def find_params(parameters):
  best_score=0
  best_params=None
  for arch in parameters['architecture']:
    for lr in parameters['lr']:
      for drop_rate in parameters['dropout']:
        for batch in parameters['batch']:
            model=create_model(arch,drop_rate,lr)
            model.fit(X_train,y_train,batch_size=batch,epochs=15,verbose=0)
            score=model.evaluate(X_val,y_val,verbose=0)[1]
            if score>best_score:
              best_score=score
              best_params={'architecture':arch,'learning_rate':lr,'dropout_rate':drop_rate,'batch_size':batch}
  return best_params

# find best hyperparameters
hyperparams=find_params(parameters)
print('Best hyperparameters are: ', hyperparams)

#build and train a model
model=create_model(hyperparams['architecture'], hyperparams['dropout_rate'], hyperparams['learning_rate'])
history=model.fit(X_train,y_train,batch_size=hyperparams['batch_size'],epochs=15,validation_data=(X_val,y_val),verbose=0)

#save model
joblib.dump(KNN, 'neural_network.pkl')

#plot training curves
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Accuracy and loss function')
plt.xlabel('Epoch')
plt.legend(['accuracy','loss'])
plt.savefig('training_plot.png')
plt.show()

### 5. Evaluate neural network and other models

# predict previously unseen test data
y_heuristic=heuristic.predict(X_test)
y_knn=KNN.predict(X_test)
y_tree=Tree.predict(X_test)
y_neural=model.predict(X_test)
y_neural=np.argmax(y_neural,axis=1)

# print accuracy scores and f1 score
print('Accuracy score for heuristic model: {}'.format(accuracy_score(y_test,y_heuristic)))
print('F1-score for heuristic model: {}'.format(f1_score(y_test,y_heuristic,average=None)))
print('Accuracy score for KNN model: {}'.format(accuracy_score(y_test,y_knn)))
print('F1-score for KNN model: {}'.format(f1_score(y_test,y_knn,average=None)))
print('Accuracy score for Decision Tree: {}'.format(accuracy_score(y_test,y_tree)))
print('F1-score for Decision Tree: {}'.format(f1_score(y_test,y_tree,average=None)))
print('Accuracy score for Neural Network: {}'.format(accuracy_score(y_test,y_neural)))
print('F1-score for Neural Network: {}'.format(f1_score(y_test,y_neural,average=None)))

