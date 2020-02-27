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
train_acc_rhc = []
test_acc_rhc = []
train_acc_sa = []
test_acc_sa = []
train_acc_ga = []
test_acc_ga = []
time_rhc = []
time_sa = []
time_ga = []
m=np.arange(3000,10000,1000)
m1=np.arange(100,800,100)
for i in range(0,7,1):
    nn_model_rhc = ml.NeuralNetwork(hidden_nodes=[50,25], activation = 'relu',
                                 algorithm = 'random_hill_climb', max_iters =np.int(m[i]),
                                 bias = True, is_classifier = True, learning_rate = 0.5,
                                 early_stopping = True, max_attempts = 100,
                                 random_state = 7)
    a=time.time()
    nn_model_rhc.fit(X_train,y_train)
    b=time.time()
    time_rhc.append(b-a)
    y_train_pred_rhc = nn_model_rhc.predict(X_train)
    y_train_accuracy_rhc = accuracy_score(y_train, y_train_pred_rhc)
    train_acc_rhc.append(y_train_accuracy_rhc)
    y_test_pred_rhc = nn_model_rhc.predict(X_test)
    y_test_accuracy_rhc = accuracy_score(y_test, y_test_pred_rhc)
    test_acc_rhc.append(y_test_accuracy_rhc)

    print('RHC ',np.int(m[i]),y_train_accuracy_rhc,y_test_accuracy_rhc)
    
    nn_model_sa = ml.NeuralNetwork(hidden_nodes = [50,25], activation = 'relu',
                                 algorithm = 'simulated_annealing', max_iters = np.int(m[i]),
                                 bias = True, is_classifier = True, learning_rate = 0.5,
                                 early_stopping = True, max_attempts = 100,
                                 random_state = 7)
    a=time.time()

    nn_model_sa.fit(X_train,y_train)
    b=time.time()
    time_sa.append(b-a)
    y_train_pred_sa = nn_model_sa.predict(X_train)
    y_train_accuracy_sa = accuracy_score(y_train, y_train_pred_sa)
    train_acc_sa.append(y_train_accuracy_sa)
    y_test_pred_sa = nn_model_sa.predict(X_test)
    y_test_accuracy_sa = accuracy_score(y_test, y_test_pred_sa)
    test_acc_sa.append(y_test_accuracy_sa)

    print('sa ',np.int(m[i]),y_train_accuracy_sa,y_test_accuracy_sa)

    nn_model_ga = ml.NeuralNetwork(hidden_nodes = [50,25], activation = 'relu',
                                 algorithm = 'genetic_alg', max_iters =np.int(m1[i]),
                                 bias = True, is_classifier = True, learning_rate = 0.9,
                                 early_stopping = True, max_attempts = 100,
                                 random_state = 7,pop_size=200,mutation_prob=.1)
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

    print('ga ',np.int(m1[i]),y_train_accuracy_ga,y_test_accuracy_ga)

plt.figure()
plt.plot(np.arange(0,7,1), np.array(test_acc_rhc), label='RHC')
plt.plot(np.arange(0,7,1), np.array(test_acc_sa), label='SA')
plt.plot(np.arange(0,7,1), np.array(test_acc_ga), label='GA')
plt.xlabel('Number of Iterations')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs. increasing Iterations for Optimization Algorithms')
plt.legend()
plt.savefig('nn_test_iterations1.png')

plt.figure()
plt.plot(np.arange(0,7,1), np.array(train_acc_rhc), label='RHC')
plt.plot(np.arange(0,7,1), np.array(train_acc_sa), label='SA')
plt.plot(np.arange(0,7,1), np.array(train_acc_ga), label='GA')
plt.xlabel('increasing Iterations')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy vs. increasing Iterations for Optimization Algorithms')
plt.legend()
plt.savefig('nn_train_iterations1.png')

plt.figure()
plt.plot(np.arange(0,7,1), np.array(time_rhc), label='RHC')
plt.plot(np.arange(0,7,1), np.array(time_sa), label='SA')
plt.plot(np.arange(0,7,1), np.array(time_ga), label='GA')
plt.xlabel('increasing Iterations')
plt.ylabel('Training Time')
plt.title('Computation Time vs. incr. Iterations for Optimization Algorithms')
plt.legend()
plt.savefig('nn_computation.png')

