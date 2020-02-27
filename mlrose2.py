import pandas as pd
from sklearn.model_selection import train_test_split
import mlrose as ml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
df = pd.read_csv('tic-tac-toe.data', sep=",", skiprows=0)

df = np.array(df)
print(df.shape, df.dtype)
dat = df[:, 0:9]
tar1 = df[:, 9]
X = dat
y = tar1
print(y.dtype)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
train_acc_rhc = []
test_acc_rhc = []
train_acc_sa = []
test_acc_sa = []
train_acc_ga = []
test_acc_ga = []
time_rhc = []
time_sa = []
time_ga = []
for r in range(0,20,2):
    nn_model_rhc = ml.NeuralNetwork(hidden_nodes=[50, 25], activation='relu',
                                    algorithm='random_hill_climb', max_iters=4000,
                                    bias=True, is_classifier=True, learning_rate=0.6,
                                    early_stopping=True, max_attempts=100,
                                    random_state=7,restarts=r)
    nn_model_rhc.fit(X_train, y_train)

    y_train_pred_rhc = nn_model_rhc.predict(X_train)
    y_train_accuracy_rhc = accuracy_score(y_train, y_train_pred_rhc)
    train_acc_rhc.append(y_train_accuracy_rhc)
    y_test_pred_rhc = nn_model_rhc.predict(X_test)
    y_test_accuracy_rhc = accuracy_score(y_test, y_test_pred_rhc)
    test_acc_rhc.append(y_test_accuracy_rhc)

    print('RHC ', r, y_train_accuracy_rhc, y_test_accuracy_rhc)


plt.figure()
plt.plot(np.arange(0,20,2), np.array(test_acc_rhc), label='RHC')
plt.xlabel('Number of random restarts')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs. no. of random restarts(neural net)')
plt.legend()
plt.savefig('nn_test_restarts.png')

plt.figure()
plt.plot(np.arange(0,20,2), np.array(train_acc_rhc), label='RHC')
plt.xlabel('Number of random restarts')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy vs.  no. of random restarts(neural net)')
plt.legend()
plt.savefig('nn_train_restarts.png')
##########

for schedule in [ml.GeomDecay(),ml.ExpDecay(),ml.ArithDecay()]:
    nn_model_sa = ml.NeuralNetwork(hidden_nodes=[50, 25], activation='relu',
                                   algorithm='simulated_annealing', max_iters=5000,
                                   bias=True, is_classifier=True, learning_rate=0.6,
                                   early_stopping=True, max_attempts=100,
                                   random_state=7,schedule=schedule)
    nn_model_sa.fit(X_train, y_train)
    y_train_pred_sa = nn_model_sa.predict(X_train)
    y_train_accuracy_sa = accuracy_score(y_train, y_train_pred_sa)
    train_acc_sa.append(y_train_accuracy_sa)
    y_test_pred_sa = nn_model_sa.predict(X_test)
    y_test_accuracy_sa = accuracy_score(y_test, y_test_pred_sa)
    test_acc_sa.append(y_test_accuracy_sa)

    print('sa ', y_train_accuracy_sa, y_test_accuracy_sa)

plt.figure()
plt.plot(['GeomDecay', 'ExpDecay', 'ArithDecay'], np.array(test_acc_sa), label='sa')
plt.xlabel('Simulated annealing with different schedules')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs. Schedule nn')
plt.legend()
plt.savefig('nn_test_sa_schedules.png')

plt.figure()
plt.plot(['GeomDecay', 'ExpDecay', 'ArithDecay'], np.array(test_acc_sa), label='sa')
plt.xlabel('Simulated annealing with different schedules')
plt.ylabel('Train Accuracy')
plt.title('Train Accuracy vs. Schedule nn')
plt.legend()
plt.savefig('nn_train_sa_schedules.png')

for p in range(50,250,50):
    nn_model_ga = ml.NeuralNetwork(hidden_nodes = [50,25], activation = 'relu',
                                 algorithm = 'genetic_alg', max_iters = 100,
                                 bias = True, is_classifier = True, learning_rate = 0.8,
                                 early_stopping = True, max_attempts = 100,
                                 random_state = 7,mutation_prob=.1,pop_size=p)
    a=time.time()
    nn_model_ga.fit(X_train,y_train)
    b=time.time()
    time_ga.append(b-a)
    y_train_pred_ga = nn_model_ga.predict(X_train)
    y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
    train_acc_ga.append(y_train_accuracy_ga)
    y_test_pred_ga = nn_model_ga.predict(X_test)
    y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
    test_acc_ga.append(y_test_accuracy_ga)

    print('ga ',p,y_train_accuracy_ga,y_test_accuracy_ga)

plt.figure()
plt.plot(np.arange(50,250,50), np.array(test_acc_ga), label='GA')
plt.xlabel('Population size')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs. Population size(neural net)')
plt.legend()
plt.savefig('nn_test_ga_size.png')

plt.figure()
plt.plot(np.arange(50,250,50), np.array(train_acc_ga), label='GA')
plt.xlabel('Population size')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy vs.  Population size(neural net)')
plt.legend()
plt.savefig('nn_train_ga_size.png')

