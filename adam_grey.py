# Implementation of ADAM from Scratch on Gisette by Shigeng Sun
# requires gisette_scale and gisette_scale.t in the same directory
# Data is from https://archive.ics.uci.edu/ml/datasets/Gisette in
# Isabelle Guyon, Steve R. Gunn, Asa Ben-Hur, Gideon Dror, 2004.
# Result analysis of the NIPS 2003 feature selection challenge. In: NIPS.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file

X_train, y_train = load_svmlight_file('gisette_scale.t')
X_test, y_test = load_svmlight_file('gisette_scale')

y_train = np.multiply(y_train==1,y_train)
y_test = np.multiply(y_test==1,y_test)

lam = 1e-3
alpha = 2**-2
Num_epoch = 1
Num_train , Num_Features = X_train.shape
Num_test , Num_Features = X_test.shape
np.arange(Num_Features)

Order_list = np.arange(Num_train)

def logloss( X , y , w , lam):
    Xw = X @ w 
    l = - np.transpose(Xw) @ y + sum( np.log( 1 + np.exp( Xw ) ) ) + lam * (np.inner(w,w)); 
    return l

def full_grad(X, y, w , lam):
    mu = 1./(1+np.exp(-X @ w)) 
    g = np.transpose(X)@(mu-y) + 2 * lam * w
    return g

def stoch_grad(X, y, w , lam , i):
    mu = 1./(1+np.exp(-X[i,:] @ w))
    g = np.transpose(X[i,:])@(mu-y[i]) + 2 * lam * w
    return g


w  = np.zeros(Num_Features)
train_accy_record = np.zeros(Num_epoch * Num_train)
train_loss_record = np.zeros(Num_epoch * Num_train)
test_accy_record = np.zeros(Num_epoch * Num_train)
test_loss_record = np.zeros(Num_epoch * Num_train)


alpha = 2**-10
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
B = 100
u = np.zeros(Num_Features)
v = np.zeros(Num_Features)



for t in range(Num_train * Num_epoch):
    indexlist = np.arange(Num_train)
    np.random.shuffle(indexlist)
    g = np.zeros(Num_Features)
    for i in range(B):
        g = g + stoch_grad(X_train , y_train , w , lam , i)
    g = g/B
    u = beta1 * u + (1-beta1) * g
    v = beta2 * v + (1-beta2) * np.multiply(g,g)
    uh = u /(1-beta1)
    vh = v /(1-beta2)
    w = w - alpha * uh /(np.sqrt(vh) + eps)
    
    
    l = logloss(X_train , y_train , w , lam)
    train_accy = sum(((X_train@w)>0 )==y_train) /Num_train
    ll = logloss(X_test , y_test , w , lam)
    test_accy = sum(((X_test@w)>0 )==y_test) /Num_test

    train_accy_record[t ] = train_accy
    train_loss_record[t] = l
    test_accy_record[t] = test_accy
    test_loss_record[t] = ll

    print(np.floor(t/(Num_train * Num_epoch)),np.floor(train_accy*100),l)
fig, (ax1, ax2,ax3,ax4) = plt.subplots(4)
ax1.plot(train_accy_record)
ax2.plot(train_loss_record)
ax3.plot(test_accy_record)
ax4.plot(test_loss_record)
plt.show()
