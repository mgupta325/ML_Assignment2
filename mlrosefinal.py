import pandas as pd
from sklearn.model_selection import train_test_split
import mlrose as ml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,balanced_accuracy_score
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
df = pd.read_csv('tic-tac-toe.data', sep=",", skiprows=0)

df=np.array(df)
print(df.shape,df.dtype)
dat=df[:,0:9]
tar1=df[:,9]
X=dat
y=tar1
print(y.dtype)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

nn_model_rhc = ml.NeuralNetwork(hidden_nodes=[50, 25], activation='relu',
                                algorithm='random_hill_climb', max_iters=6000,
                                bias=True, is_classifier=True, learning_rate=0.6,
                                early_stopping=True, max_attempts=100,
                                random_state=7, restarts=12)
a=time.time()
nn_model_rhc.fit(X_train, y_train)
b=time.time()
print('training time',b-a)
y_train_pred_rhc = nn_model_rhc.predict(X_train)
y_train_accuracy_rhc = accuracy_score(y_train, y_train_pred_rhc)
y_train_accuracy_rhc1 = balanced_accuracy_score(y_train, y_train_pred_rhc)
print(confusion_matrix(y_train, y_train_pred_rhc),y_train_accuracy_rhc,y_train_accuracy_rhc1)

y_test_pred_rhc = nn_model_rhc.predict(X_test)
y_test_accuracy_rhc = accuracy_score(y_test, y_test_pred_rhc)
y_test_accuracy_rhc1 = balanced_accuracy_score(y_test, y_test_pred_rhc)
print(confusion_matrix(y_test, y_test_pred_rhc),y_test_accuracy_rhc,y_test_accuracy_rhc1)
print('rhc ')


nn_model_sa = ml.NeuralNetwork(hidden_nodes=[50, 25], activation='relu',
                               algorithm='simulated_annealing', max_iters=8000,
                               bias=True, is_classifier=True, learning_rate=0.6,
                               early_stopping=True, max_attempts=100,
                               random_state=7, schedule=ml.ExpDecay())
a=time.time()
nn_model_sa.fit(X_train, y_train)
b=time.time()
print('training time',b-a)
y_train_pred_sa = nn_model_sa.predict(X_train)
y_train_accuracy_sa = accuracy_score(y_train, y_train_pred_sa)
y_train_accuracy_sa1 = balanced_accuracy_score(y_train, y_train_pred_sa)
print(confusion_matrix(y_train, y_train_pred_sa),y_train_accuracy_sa,y_train_accuracy_sa1)

y_test_pred_sa = nn_model_sa.predict(X_test)
y_test_accuracy_sa = accuracy_score(y_test, y_test_pred_sa)
y_test_accuracy_sa1 = balanced_accuracy_score(y_test, y_test_pred_sa)
print(confusion_matrix(y_test, y_test_pred_sa),y_test_accuracy_sa,y_test_accuracy_sa1)
print('sa ')

nn_model_ga = ml.NeuralNetwork(hidden_nodes=[50, 25], activation='relu',
                               algorithm='genetic_alg', max_iters=100,
                               bias=True, is_classifier=True, learning_rate=0.9,
                               early_stopping=True, max_attempts=100,
                               random_state=7, mutation_prob=.1, pop_size=250)
a=time.time()
nn_model_ga.fit(X_train, y_train)
b=time.time()
print('training time',b-a)

y_train_pred_ga = nn_model_ga.predict(X_train)
y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
y_train_accuracy_ga1 = balanced_accuracy_score(y_train, y_train_pred_ga)
print(confusion_matrix(y_train, y_train_pred_ga),y_train_accuracy_ga,y_train_accuracy_ga1)

y_test_pred_ga = nn_model_ga.predict(X_test)
y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
y_test_accuracy_ga1 = balanced_accuracy_score(y_test, y_test_pred_ga)
print(confusion_matrix(y_test, y_test_pred_ga),y_test_accuracy_ga,y_test_accuracy_ga1)
print('ga')