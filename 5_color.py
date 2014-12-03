import timeit
from multiprocessing import Pool

import cv2
import numpy as np

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model.logistic import LogisticRegression

from utils import *

import warnings
warnings.filterwarnings("ignore")

TEST = False
SAVE = True
PREFIX = "COLOR"

train_images, train_labels, test_images, test_labels = get_train_test(TEST)

def classify_svm(train_features, train_labels, test_features):
    global SAVE
    clf = svm.SVC(C = 0.005, kernel = 'linear', )
    clf.fit(train_features, train_labels)

    if not TEST and SAVE:
        save_pickle(PREFIX+"_svm", clf)

    return clf.predict(test_features)

def classify_logistic(train_features, train_labels, test_features):
    global SAVE
    clf = LogisticRegression()
    clf.fit(train_features, train_labels)

    if not TEST and SAVE:
        save_pickle(PREFIX+"_logistic", clf)

    return clf.predict(test_features)

pool = Pool(cv2.getNumberOfCPUs())

train_hist = pool.map(get_color_histogram, train_images)
test_hist = pool.map(get_color_histogram, test_images)

print "\n ["+PREFIX+"][*] Classifying SVM"
result = []
start = timeit.default_timer()
pred = classify_svm(train_hist, train_labels, test_hist)
stop = timeit.default_timer()

print " ["+PREFIX+"][=] SVM time : %s" % (stop - start)

correct = sum(1.0*(pred == test_labels))
accuracy = correct / len(test_labels)
result.append(str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(test_labels))+ ")")
print " ["+PREFIX+"][=] SVM result : ", "\n".join(result)

print "\n ["+PREFIX+"][*] Classifying Regression"
result = []
start = timeit.default_timer()
pred = classify_logistic(train_hist, train_labels, test_hist)
stop = timeit.default_timer()
print " ["+PREFIX+"][=] LR time : %s" % (stop - start)

correct = sum(1.0*(pred == test_labels))
accuracy = correct / len(test_labels)
result.append(str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(test_labels))+ ")")
print " ["+PREFIX+"][=] LR result :", "\n".join(result)

if not TEST:
    send_mail("Food finished", "<br/>".join(result))
