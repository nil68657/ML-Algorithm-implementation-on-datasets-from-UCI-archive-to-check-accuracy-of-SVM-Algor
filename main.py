from Utils import Utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score as cvs


def optical_svm(training_file, test_file):
    training = Utils.extract_optical_xls_file(training_file)
    test = Utils.extract_optical_xls_file(test_file)
    clf = SVC(C=100, kernel="linear")
    clf.fit(training['data'], training['result'])
    print(clf.score(test['data'], test['result']))
    print(cvs(clf, X=test['data'], y=test['result'], verbose=1,cv=5))


def amazon_svm(training_file, test_file):
    training = Utils.extract_amazon_xls_file(training_file)
    test = Utils.extract_amazon_xls_file(test_file)
    clf = SVC(C=200, kernel="rbf")
    clf.fit(training['data'], training['result'])
    print(clf.score(test['data'], test['result']))
    print(cvs(clf, X=test['data'], y=test['result'], verbose=1, cv=5))

optical_svm('./o_train.csv', './o_test.csv')
amazon_svm('./a_train.csv', './a_test.csv')
