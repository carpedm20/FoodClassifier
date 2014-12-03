import timeit
import itertools
from math import sqrt
from multiprocessing import Pool

import cv2
import numpy as np

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model.logistic import LogisticRegression

from utils import *

import warnings
warnings.filterwarnings("ignore")

TEST = False
SAVE = False
LOWE = False

train_images, train_labels, test_images, test_labels = get_train_test(TEST)

pool = Pool(cv2.getNumberOfCPUs())
if LOWE:
    print " [!] Lowe's SIFT"
    train_sift_with_null = pool.map(get_sift_lowe, train_images)
    test_sift_with_null = pool.map(get_sift_lowe, test_images)
else:
    print " [!] OpenCV2's SIFT"
    train_sift_with_null = pool.map(get_sift, train_images)
    test_sift_with_null = pool.map(get_sift, test_images)
pool.terminate()                                                                                        

train_sift = removing_null(train_sift_with_null, train_labels)
reduced_train_sift = np.concatenate(train_sift, axis = 0)

test_sift = removing_null(test_sift_with_null, test_labels)
reduced_test_sift = np.concatenate(test_sift, axis = 0)

print "\n [*] Kmeans fitting"
start = timeit.default_timer()
k = 1000

nfeatures = reduced_train_sift.shape[0]
k = int(sqrt(nfeatures))

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

all_descriptors = np.concatenate((reduced_train_sift,reduced_test_sift), axis=0)

kmeans.fit(all_descriptors)
stop = timeit.default_timer()

print " => Kmeans time : %s" % (stop - start)
if not TEST and SAVE:
    save_pickle("kmeans",kmeans)

print "\n [*] Spatial Pyramid Histogram calculation"
pool = Pool(cv2.getNumberOfCPUs()-2)

#l=[]
#for image in train_images:
#    l.append(get_spatial_pyramid((image, kmeans.cluster_centers_, reduced_train_sift.shape[0])))

tmp = itertools.repeat(kmeans.cluster_centers_)
tmp2 = itertools.repeat(all_descriptors.shape[0])
train_spp_hist = pool.map(get_spatial_pyramid, itertools.izip(train_images, tmp, tmp2))

tmp = itertools.repeat(kmeans.cluster_centers_)
tmp2 = itertools.repeat(all_descriptors.shape[0])
test_spp_hist = pool.map(get_spatial_pyramid, itertools.izip(test_images, tmp, tmp2))

pool.terminate()                                                                                        
def classify_svm(train_features, train_labels, test_features):
    global SAVE
    #clf = svm.SVC(C = 0.005, kernel = 'linear', )
    #clf = svm.SVC(C = 0.005, kernel = 'rbf', )
    clf = OneVsRestClassifier(svm.LinearSVC())
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


print "\n [*] Classifying SVM"
result = []
start = timeit.default_timer()
pred = classify_svm(train_spp_hist, train_labels, test_spp_hist)
stop = timeit.default_timer()

print " [=] SVM time : %s" % (stop - start)

correct = sum(1.0*(pred == test_labels))
accuracy = correct / len(test_labels)
result.append(str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(test_labels))+ ")")
print " [=] SVM result : ", "\n".join(result)

print "\n [*] Classifying Regression"
result = []
start = timeit.default_timer()
pred = classify_logistic(train_spp_hist, train_labels, test_spp_hist)
stop = timeit.default_timer()
print " [=] LR time : %s" % (stop - start)

correct = sum(1.0*(pred == test_labels))
accuracy = correct / len(test_labels)
result.append(str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(test_labels))+ ")")
print " [=] LR result :", "\n".join(result)

if not TEST:
    send_mail("Food finished", "<br/>".join(result))
