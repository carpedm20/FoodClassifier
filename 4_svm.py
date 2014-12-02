from random import shuffle
import numpy as np
import cv2

from skimage.io import imread
from skimage import color, feature
from skimage.transform import resize

from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import svm

from glob import glob
from multiprocessing import Pool

from caffe_io import resize_image, oversample

TEST = False
FOOD_PATH = "/home/carpedm20/data/food100/"

SINGLE_FOOD = "/home/carpedm20/data/food100/%s/crop_*.jpg"

if TEST:
    food1 = glob(SINGLE_FOOD % '1')
    print "\nfood1 : %s" % len(food1)
    food2 = glob(SINGLE_FOOD % '36')
    print "food2 : %s" % len(food2)
    food3 = glob(SINGLE_FOOD % '23')
    print "food3 : %s" % len(food3)
    foods = food1 + food2 + food3
else:
    foods = glob("/home/carpedm20/data/food100/*/crop_*.jpg")

def build_labels(foods):
    new_foods = []
    for food in foods:
        food_label = food[len(FOOD_PATH):].split("/")[0]
        new_foods.append((food, food_label))
    return new_foods

foods = build_labels(foods)

train, test = train_test_split(foods, test_size=0.33, random_state=42)

shuffle(train)
shuffle(test)

if TEST:
    train = train[:len(train)/50]
    test = test[:len(test)/50]

print "\ntrain : %s" % len(train)
print "test : %s" % len(test)

train_images = [x[0] for x in train]
train_labels = [int(x[1]) for x in train]

test_images = [x[0] for x in test]
test_labels = [int(x[1]) for x in test]

print "\n [*] Ready for learning"

def get_sift(img):
    raw = cv2.imread(img)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, desc = sift.detectAndCompute(gray, None)

    return desc

def get_hist_feature(sift_features, predicted_labels):
    feature_num = [f.shape[0] for f in sift_features]
    hist = np.zeros(shape = (len(feature_num), 1000))
    for i, num in enumerate(feature_num):
        labels = predicted_labels[:num]
        for label in labels:
            hist[i, label] = hist[i, label] + 1
        predicted_labels = predicted_labels[num:]
    return hist

def get_histogram(k, feature_list, predicted_labels):
    hist = np.zeros(shape = (len(feature_list), k))
    for i, feature in enumerate(feature_list):
        current_hist, bins = np.histogram(feature, bins=k)
        hist[i] = current_hist
    return hist

def classify_svm(train_features, train_labels, test_features):
    clf = svm.SVC(C = 0.005, kernel = 'linear', )
    clf.fit(train_features, train_labels)

    return clf.predict(test_features)

def classify_logistic(train_features, train_labels, test_features):
    clf = LogisticRegression()
    clf.fit(train_features, train_labels)

    return clf.predict(test_features)

def reduce_sift(mapping):
    return np.concatenate(train_sift, axis = 0)
    #return reduce(lambda x, y: np.concatenate((x, y), axis = 0), mapping)
    a = mapping[0]
    for i in mapping[1:]:
        a=np.concatenate((a,i),axis=0)
    return a

print "\n [*] Creating process pool"

pool = Pool(cv2.getNumberOfCPUs())

print "\n [*] Training shift"
train_sift = pool.map(get_sift, train_images)
train_sift = [i for i in train_sift if i != None]
print "\n [*] Reducing training shift"
reduced_train_sift = np.concatenate(train_sift, axis = 0)

print "\n [*] Testing shift"
test_sift = pool.map(get_sift, test_images)
test_sift = [i for i in test_sift if i != None]
print "\n [*] Reducing testing shift"
reduced_test_sift = np.concatenate(test_sift, axis = 0)

print "\n [*] Kmeans fitting"
k = 1000
kmeans = MiniBatchKMeans(n_clusters = k, batch_size = 1000, max_iter = 250)
kmeans.fit(reduced_train_sift)

print "\n [*] Predicting sift"
train_predicted = kmeans.predict(reduced_train_sift)
test_predicted = kmeans.predict(reduced_test_sift)

print "\n [*] Creating histogram of sift"
train_hist_features = get_histogram(k, train_sift, train_predicted)
test_hist_features = get_histogram(k, test_sift, test_predicted)

pred = classify_svm(train_hist_features, train_labels, test_hist_features)

result = []

print "\n [*] Classifying SVM"
correct = sum(1.0*(pred == test_labels))
accuracy = correct / len(test_labels)
result.append("SVM : " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(test_labels))+ ")")

pred = classify_logistic(train_hist_features, train_labels, test_hist_features)
print "\n".join(result)

print "\n [*] Classifying Regression"
correct = sum(1.0*(pred == test_labels))
accuracy = correct / len(test_labels)
result.append("Logistic Regression : " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(test_labels))+ ")")
print "\n".join(result)

from utils import *

send_mail("Food finished", "<br/>".join(result))

#for test_feature, label in zip(test_features, test_labels):
#    predict = classifier.predict(test_features)
#    print "Real : %s, Predict : %s" % (label, predict)

