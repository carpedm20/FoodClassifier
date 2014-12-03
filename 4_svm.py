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

train_images, train_labels, test_images, test_labels = get_train_test(TEST)

def classify_svm(train_features, train_labels, test_features):
    global SAVE
    clf = svm.SVC(C = 0.005, kernel = 'linear', )
    clf.fit(train_features, train_labels)

    if not TEST and SAVE:
        save_pickle("svm", clf)

    return clf.predict(test_features)

def classify_logistic(train_features, train_labels, test_features):
    global SAVE
    clf = LogisticRegression()
    clf.fit(train_features, train_labels)

    if not TEST and SAVE:
        save_pickle("logistic", clf)

    return clf.predict(test_features)

pool = Pool(cv2.getNumberOfCPUs())

train_sift_with_null = pool.map(get_sift, train_images)
train_sift = removing_null(train_sift_with_null, train_labels)
reduced_train_sift = np.concatenate(train_sift, axis = 0)

test_sift_with_null = pool.map(get_sift, test_images)
test_sift = removing_null(test_sift_with_null, test_labels)
reduced_test_sift = np.concatenate(test_sift, axis = 0)

print "\n [*] Kmeans fitting"
start = timeit.default_timer()
k = 1000

if False:
    kmeans = KMeans(n_clusters = k,
                    init       ='k-means++',
                    n_init     = 10,
                    max_iter   = 100,
                    n_jobs     = -1)
else:
    kmeans = MiniBatchKMeans(n_clusters = k,
                             init       ='k-means++',
                             n_init     = 10,
                             max_iter   = 100,
                             init_size  = 1000,
                             batch_size = 1000)

kmeans.fit(reduced_train_sift)
stop = timeit.default_timer()

print " => Kmeans time : %s" % (stop - start)
if not TEST and SAVE:
    save_pickle("kmeans",kmeans)

start = timeit.default_timer()
train_predicted = kmeans.predict(reduced_train_sift)
test_predicted = kmeans.predict(reduced_test_sift)
stop = timeit.default_timer()

print "\n [*] Creating histogram of sift"
train_hist_features = get_histogram(k, train_sift, train_predicted)
test_hist_features = get_histogram(k, test_sift, test_predicted)

print "\n [*] Classifying SVM"
result = []
start = timeit.default_timer()
pred = classify_svm(train_hist_features, train_labels, test_hist_features)
stop = timeit.default_timer()

print " [=] SVM time : %s" % (stop - start)

correct = sum(1.0*(pred == test_labels))
accuracy = correct / len(test_labels)
result.append(str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(test_labels))+ ")")
print " [=] SVM result : ", "\n".join(result)

print "\n [*] Classifying Regression"
result = []
start = timeit.default_timer()
pred = classify_logistic(train_hist_features, train_labels, test_hist_features)
stop = timeit.default_timer()
print " [=] LR time : %s" % (stop - start)

correct = sum(1.0*(pred == test_labels))
accuracy = correct / len(test_labels)
result.append(str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(test_labels))+ ")")
print " [=] LR result :", "\n".join(result)

if not TEST:
    send_mail("Food finished", "<br/>".join(result))
