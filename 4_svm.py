from random import shuffle

import numpy as np
import mahotas as mh

import cv2

from skimage.io import imread
from skimage import color, feature, filter
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

TEST = True
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
    foods = glob.glob("/home/carpedm20/data/food100/*/crop_*.jpg")

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
    test = test[:len(train)/50]

print "\ntrain : %s" % len(train)
print "test : %s" % len(test)

train_images = [x[0] for x in train]
train_labels = [int(x[1]) for x in train]

test_images = [x[0] for x in test]
test_labels = [int(x[1]) for x in test]

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
    feature_num = [f.shape[0] for f in feature_list]
    hist = np.zeros(shape = (len(feature_num), k))
    for i, num in enumerate(feature_num):
        labels = predicted_labels[:num]
        for label in labels:
            hist[i, label] = hist[i, label] + 1
        predicted_labels = predicted_labels[num:]
    return hist

def reduce_sift(mapping):
    return reduce(lambda x, y: np.concatenate((x, y), axis = 0), mapping)

def classify(train_features, train_labels, test_features):
    clf = svm.SVC(C = 0.005, kernel = 'linear', )
    clf.fit(train_features, train_labels)
    #clf = LogisticRegression()
    #clf.fit(train_features, train_labels)
    predicted_labels = clf.predict(test_features)
    return predicted_labels

def main():
    p = Pool(cv2.getNumberOfCPUs())

    train_sift = reduce_sift(p.map(get_sift, train_images))
    test_sift = reduce_sift(p.map(get_sift, train_images))

    k = 1000
    kmeans = MiniBatchKMeans(n_clusters = k, batch_size = 1000, max_iter = 250)
    kmeans.fit(train_sift)

    train_predicted = kmeans.predict(train_sift)
    test_predicted = kmeans.predict(test_sift)

    train_hist_features = get_histogram(k, train_sift, train_predicted)
    test_hist_features = get_histogram(k, test_sift, test_predicted)

    pred = classify(train_hist_features, train_labels, test_hist_features)
    out = pd.DataFrame(pred, columns = ['label'])
    out = out.astype(int)
    out.index += 1
    out.to_csv('sub1.csv', index_label = 'id')

    #for test_feature, label in zip(test_features, test_labels):
    #    predict = classifier.predict(test_features)
    #    print "Real : %s, Predict : %s" % (label, predict)

if __name__ == "__main__":
    main()
