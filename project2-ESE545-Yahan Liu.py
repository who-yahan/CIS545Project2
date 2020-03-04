# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
##################################problem1
import numpy as np
from sklearn.datasets import fetch_rcv1
import matplotlib.pyplot as plt

rcv1 = fetch_rcv1()

y = rcv1['target'][:,33].toarray()
y = y.astype(float)
for i in range(0,len(y)):
    if  y[i] != 1:
        y[i] = -1
        
X = rcv1['data']   

#X_train=X[:100000]
#X_test=X[100000:]
#y_train=y[:100000]
#y_test=y[100000:]
X_train=X[:10000]
X_test=X[10000:]
y_train=y[:10000]
y_test=y[10000:]
   
(N,k) = X_train.shape
print(N,k)
(a,b) = y_train.shape
print(a,b)

#%%
################################probelm2
#initialization procedure
def pegasos(X_train, y_train, T, B, lamda):
    '''Pegasos Algorithm SVM
    @param T is iteration number
    @param B is batch size
    
    Returns weights w and learning accuracy list accuracy
    '''
    (N,k) = X_train.shape
    w = np.zeros((k,1)) 
    accuracy=[]
    for t in range(T):
        batch_index = np.random.randint(N,size=B)
        #this is a (sparse) matrix of shape (B,k)
        batch = X_train[batch_index,:]
        batch_labels = y_train[batch_index] #a (B,1) vector of classification labels
        X_w_dot = batch.dot(w).reshape(B,1)#batch is sparse matrix, w is numpy array
        #print(X_w_dot, batch_labels) #batch_labels is array
        tests = X_w_dot * batch_labels#this is a (B,1) vector to compare with 1 
        #print(tests.shape)
        new_batch_index = []
        for i,test in enumerate(tests):
            if tests[i] <1:
                new_batch_index.append(i)
        reduced_batch = batch[new_batch_index,:]#this is a matrix of shape (new_batch_index, k)
        #print(reduced_batch)
        reduced_batch_labels = batch_labels[new_batch_index]#this is a vector of (new_batch_index, 1)
        #print(reduced_batch_labels)
        reduced_batch = reduced_batch.transpose()
        #print(reduced_batch.shape)
        #print(reduced_batch_labels.shape)
        grad = lamda*w-1/float(B)*reduced_batch.dot(reduced_batch_labels)
        step_size = 1/(lamda*(t+1)) #also known as the learning rate
        w = w-step_size*grad #gradient descent
        w = min(1,1/(np.linalg.norm(w)*np.sqrt(lamda)))*w
        y_pred=np.sign(X_train*w)
        right=np.where(y_train==y_pred)[0]
        accuracy.append(len(right)/y_train.shape[0])
    return w,accuracy, y_pred, y_train

w,accuracy, y_pred, y_train = pegasos(X_train, y_train, 100, 100, 0.001)
accuracy=accuracy[-10:]
    

plt.figure()
plt.plot(accuracy)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Pegasos(B=100,lamda=0.001)')
plt.show()



lamda_space = 10**np.linspace(-1,-3,num=3)
B_space = np.linspace(100,900,num=3,dtype=int)
T = [100, 20, 100/9]
error = np.zeros((len(lamda_space),len(B_space)))
for n,lamda in enumerate(lamda_space):
    for i,B in enumerate(B_space):
        w_t,accuracy = pegasos(X_train,y_train,T[i],B,lamda)
        error[n,i] = 1-accuracy[-1]

plt.figure()
for n,lamda in enumerate(lamda_space):
    plt.plot(B_space,error[n,:],label=str(lamda))
plt.xlabel('Batch size')
plt.ylabel('Error')
plt.title('PEGASOS training error given batch size for T=100')
plt.legend()
#%%
##############################problem3
#def adagrad(X_train,y_train,T,B,lamda):
#    (N,k) = X_train.shape
#    s = np.ones(k).reshape(k,1)
#    eta = 1/np.sqrt(T)
#    w = np.zeros((k,1))
#    accuracy = []
#    for t in range(T):
#        #indices = iindices[t]
#        batch_index = np.random.randint(N,size=B)
#        batch = X_train[batch_index,:] #this is a (sparse) matrix of shape (B,k) with rows feature vectors
#        batch_labels = y_train[batch_index] #a (B,1) vector of classification labels
#        X_w_dot = batch.dot(w).reshape(B,1)#batch is sparse matrix, w is matrix
#        #print(batch_labels.shape) batch_labels is array
#        tests = X_w_dot * batch_labels#this is a (B,1) vector to compare with 1 
#        new_batch_index = []
#        for index,test in enumerate(tests):
#            if tests[index] <1:
#                new_batch_index.append(index)
#        reduced_batch = batch[new_batch_index,:]
#        reduced_batch_labels = batch_labels[new_batch_index]
#        reduced_batch = reduced_batch.transpose()
#        #print(reduced_batch.shape)
#        #print(reduced_batch_labels.shape)
#        w = w.reshape(k,1)
#        grad = lamda*w-1/float(B)*reduced_batch.dot(reduced_batch_labels)
##        print(type(grad))
##        print(grad.shape)
##        a = grad*grad
##        print(type(a))
##        print(a.shape)
##        print(s.shape)
#        s += grad**2
#        learning_rate = eta/np.sqrt(s) #also known as the step size
#        w = w-learning_rate*grad #gradient descent
#        w = min(1,1/(np.linalg.norm(w)*np.sqrt(lamda)))*w #projection
#        y_pred=np.sign(X_train*w)
#        right=np.where(y_train==y_pred)[0]
#        accuracy.append(len(right)/y_train.shape[0])
#    return w,accuracy
#
##w3,accuracy3=adagrad(X_train, y_train, 100, 100, 0.001)
##    
##import matplotlib.pyplot as plt
##plt.figure()
##plt.plot(accuracy3)
##plt.xlabel('Iterations')
##plt.ylabel('Accuracy')
##plt.title('AdaGrad(B=100, lambda=0.001)')
##
##
##%%
###################problem4
#############part1
#from keras.models import Sequential
#
##Import 'Dense' from 'keras.layers'
#from keras.layers import Dense
#from keras.optimizers import SGD 
#
#
#
#y_train=(y_train+1)/2
#
#
##Initailozize the constructor (1 hiden layers)
#model1 = Sequential()
#model1.add(Dense(units=100, activation='relu',input_dim = 47236))
#model1.add(Dense(1, activation='sigmoid'))
#sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
#model1.compile(optimizer= sgd,
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#
##Train the model(1 hidden layers)
#history1 = model1.fit(X_train, y_train, batch_size=100, epochs = 5)
##Trianing error for 1 hidden layers
#hist_plot1 = [1-i for i in history1.history['acc'] ]
#plt.plot(hist_plot1,label='one hidden layer')
#plt.title('Error')
#plt.legend()

#################################################################

##Initialize the constructor
#model = Sequential()
#
##Add an input layer
#model.add(Dense(units = 100, activation='relu',input_dim = 47236))
#
##Add one hidden layer
#model.add(Dense(100, activation='relu'))
#
##Add an output layer
#model.add(Dense(1, activation='sigmoid'))
#sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
#model.compile(loss='binary_crossentropy', optimizer= sgd, metrics=['accuracy'])
#
#history = model.fit(X_train,y_train,batch_size=128,epochs=5)
#hist_plot = [1-i for i in history.history['acc']]
#plt.plot(hist_plot,label='two hidden layers')
#plt.title('Error')
#plt.legend()

##############################################################
#
##Initailozize the constructor (3 hiden layers)
#model2 = Sequential()
#model2.add(Dense(units=100, activation='relu',input_dim = 47236))
#model2.add(Dense(units=100, activation='relu') )
#model2.add(Dense(units=100, activation='relu') )
#model2.add(Dense(1, activation='sigmoid'))
#sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
#model2.compile(optimizer= sgd,
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#
#
##Train the model(3 hidden layers)
#history2 = model2.fit(X_train, y_train, batch_size=100, epochs = 5)
##Trianing error for 3 hidden layers
#hist_plot2 = [1-i for i in history2.history['acc'] ]
#plt.plot(hist_plot2,label='three hidden layers')
#plt.title('Error')
#plt.legend()
#
################part2
#
#count_add = 0
#table = []
#n_epoch = 5
#for i in range(5):
#    a = 80 + count_add
#    #Trial Several
#    model4 = Sequential()
#    model4.add(Dense(units=a, activation='relu',input_dim = 47236))
#    model4.add(Dense(units=a, activation='relu') )
#    model4.add(Dense(units=a, activation='relu') )
#    model4.add(Dense(units=a, activation='relu') )
#    model4.add(Dense(units=a, activation='relu') )
#    model4.add(Dense(units=a, activation='relu') )
#    model4.add(Dense(1, activation='sigmoid'))
#    sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
#    model4.compile(optimizer= sgd,
#                  loss='binary_crossentropy',
#                  metrics=['accuracy'])
#    #Train the model (several)
#    history4 = model4.fit(X_train,y_train,batch_size=100,epochs = n_epoch)
#    hist_plot4 = [1-i for i in history4.history['acc'] ]
#    table.append((a,hist_plot4[n_epoch -1]))
#    count_add = count_add + 10;
#print(table)
#
##%%
##problem 5
#w4,accuracy4=pegasos(X_test, y_test, 100, 500, 0.001)
#w5,accuracy5=adagrad(X_test, y_test, 100, 500, 0.001)
#print(accuracy4,accuracy5)
##Initailozize the constructor (3 hiden layers)
#model2 = Sequential()
#model2.add(Dense(units=100, activation='relu',input_dim = 47236))
#model2.add(Dense(units=100, activation='relu') )
#model2.add(Dense(units=100, activation='relu') )
#model2.add(Dense(1, activation='sigmoid'))
#sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
#model2.compile(optimizer= sgd,
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#
#
##Train the model(3 hidden layers)
#history2 = model2.fit(X_test, y_test, batch_size=100, epochs = 5)
##Trianing error for 3 hidden layers
#hist_plot2 = [1-i for i in history2.history['acc'] ]
#plt.plot(hist_plot2,label='three hidden layers')
#plt.title('Error')
#plt.legend()


