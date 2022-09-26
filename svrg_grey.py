# Implementation of SVRG from Scratch on Gisette by Shigeng Sun
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

lam = 1e-3
alpha = 2**-15
Num_epoch = 8
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
numbers = Num_epoch * Num_train
train_accy_record = np.zeros(numbers)
train_loss_record = np.zeros(numbers)
test_accy_record = np.zeros(numbers)
test_loss_record = np.zeros(numbers)


#################### SVRG #######################
s_lim = 8 # how many times evaluate the gradient overall
svrg_m = np.int(Num_train/(s_lim+1)) # updating frequency
train_record = np.zeros(Num_epoch * Num_train)
loss_record  = np.zeros(Num_epoch * Num_train)
svrg_progress_counter = 0
for s in range(s_lim):
    full_g = full_grad(X_train , y_train , w, lam)/Num_train; 
    indexlist = np.arange(Num_train)
    np.random.shuffle(indexlist)
    w_old = w

    for iii in range( np.int(Num_train * Num_epoch / s_lim)  ):
        #sample_index = np.mod(iii , Num_train )#
        sample_index = np.random.choice(Num_train)
        #sample_index = indexlist[sample_index]
        stoc_g = stoch_grad(X_train , y_train , w , lam , sample_index); 
        stoc_g_old = stoch_grad(X_train , y_train , w_old , lam , sample_index); 
        w = w - alpha * (stoc_g - stoc_g_old + full_g)
        l = logloss(X_train , y_train , w , lam)
        train_accy = sum(((X_train@w)>0 )==y_train) /Num_train
        ll = logloss(X_test , y_test , w , lam)
        test_accy = sum(((X_test@w)>0 )==y_test) /Num_test

        train_accy_record[svrg_progress_counter ] = train_accy
        train_loss_record[svrg_progress_counter] = l
        test_accy_record[svrg_progress_counter] = test_accy
        test_loss_record[svrg_progress_counter] = ll
        svrg_progress_counter = svrg_progress_counter + 1 
        print(np.floor(svrg_progress_counter/Num_epoch/Num_train*100),(100* train_accy) , l)
fig, (ax1, ax2,ax3,ax4) = plt.subplots(4)
ax1.plot(train_accy_record)
ax2.plot(train_loss_record)
ax3.plot(test_accy_record)
ax4.plot(test_loss_record)
plt.show()
    
