# Implementation of SGD from Scratch on Gisette by Shigeng Sun
# requires gisette_scale and gisette_scale.t in the same directory,
# avaliable for download from 
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
# also requires load_svmlight_file from the same libsvm library

# Data is originaly from https://archive.ics.uci.edu/ml/datasets/Gisette in
# Isabelle Guyon, Steve R. Gunn, Asa Ben-Hur, Gideon Dror, 2004.
# Result analysis of the NIPS 2003 feature selection challenge. In: NIPS.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file

X_train, y_train = load_svmlight_file('gisette_scale.t')
X_test, y_test = load_svmlight_file('gisette_scale')

y_train = np.multiply(y_train==1,y_train)
y_test = np.multiply(y_test==1,y_test)

print(y_train)

lam = 1e-3
alpha = 1e-3
Num_epoch = 2
Num_train , Num_Features = X_train.shape
Num_test , Num_Features = X_test.shape

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
#l = logloss(X_train , y_train , w , lam)
#g = full_grad(X_train , y_train , w, lam)
#gg = stoch_grad(X_train , y_train , w , lam , 3)
#print(l)

train_accy_record = np.zeros(Num_epoch * Num_train)
train_loss_record = np.zeros(Num_epoch * Num_train)
test_accy_record = np.zeros(Num_epoch * Num_train)
test_loss_record = np.zeros(Num_epoch * Num_train)
for epoch in range(Num_epoch):
    sgd_progress_counter = 0
    slist = np.arange(Num_train)
    np.random.shuffle(slist)
    for sgd_i in slist:
        #full_g = full_grad(X_train , y_train , w, lam); 
        stoc_g = stoch_grad(X_train , y_train , w , lam , sgd_i); 
        w = w - alpha * stoc_g 
        l = logloss(X_train , y_train , w , lam)
        train_accy = sum(((X_train@w)>0 )==y_train) /Num_train
        ll = logloss(X_test , y_test , w , lam)
        test_accy = sum(((X_test@w)>0 )==y_test) /Num_test
        print(epoch, np.floor(sgd_progress_counter / Num_train*100) ,np.floor(100* train_accy) , np.floor(l))
        train_accy_record[sgd_progress_counter + epoch * Num_train] = train_accy
        train_loss_record[sgd_progress_counter + epoch * Num_train] = l
        test_accy_record[sgd_progress_counter + epoch * Num_train] = test_accy
        test_loss_record[sgd_progress_counter + epoch * Num_train] = ll
        sgd_progress_counter = sgd_progress_counter +1 
fig, (ax1, ax2,ax3,ax4) = plt.subplots(4)
ax1.plot(train_accy_record)
ax2.plot(train_loss_record)
ax3.plot(test_accy_record)
ax4.plot(test_loss_record)
plt.show()
