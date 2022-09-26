# gisette-optim
Comparison of some simple Machine Learning optimization algorithms on gisette data (of classification of hand-written letters 4,9)

Data is from 
https://archive.ics.uci.edu/ml/datasets/Gisette
This data set belongs to the orignal authors, see page for more details

For this implementation, obtained data directly from 
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
and loaded data with libsvm utility load_svmlight_file.

This implementation includes Manual Implementation of

SGD,

ADAM,

SVRG

on a Logistic Regression Model, with L-2 Regularization.
