import numpy as np
import os
import cv2
import math
import cPickle
from os.path import exists, isdir, basename, join, splitext

from glob import glob
from random import shuffle
from time import gmtime, strftime

from skimage.io import imread
from sklearn.cross_validation import train_test_split

from scipy.cluster.vq import vq

import sift

def save_pickle(file_name, obj):
    #file_name = "%s_%s.pkl" % (file_name, strftime("%m%d-%H%M", gmtime()))
    with open(file_name, "wb") as f:
        cPickle.dump(obj, f)

def load_pickle(file_name):
    #file_name = "%s_%s.pkl" % (file_name, strftime("%m%d-%H%M", gmtime()))
    with open(file_name, "rb") as f:
        return cPickle.load(f)

def get_color_histogram(img):
    raw = cv2.imread(img)
    hists = []
    color = ('b','g','r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([raw],[i],None,[256],[0,255])
        #hist = cv2.calcHist([raw],[i],None,[4],[0,255])
        cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
        hists.append(np.int32(np.around(hist)).reshape((len(hist),)))
    return np.concatenate(hists, axis = 0)

def get_spatial_pyramid(args, level=2):
    """
    Code is based on https://github.com/wihoho/Image-Recognition/blob/6ef9159abdc8a282629f47761cefcf0c6b843184/Utility.py
    """
    img, cluster_centers, num_of_descriptors = args

    raw = cv2.imread(img)                                                                               
    width = raw.shape[1]
    height = raw.shape[0]

    w_step = math.ceil(width/4.0)
    h_step = math.ceil(height/4.0)

    keypoints, descriptors = get_sift(img, True)

    histogramOfLevelTwo = np.zeros((16, num_of_descriptors), dtype=np.uint8)
    for (keypoint, feature) in zip(keypoints, descriptors):
        x = keypoint.pt[0]
        y = keypoint.pt[1]
        boundaryIndex = int(x / w_step)  + int(y / h_step) *4

        shape = feature.shape[0]
        feature = feature.reshape(1, shape)

        codes, distance = vq(feature, cluster_centers)
        histogramOfLevelTwo[boundaryIndex][codes[0]] += 1

    # level 1, based on histograms generated on level two
    histogramOfLevelOne = np.zeros((4, num_of_descriptors), dtype=np.uint8)
    histogramOfLevelOne[0] = histogramOfLevelTwo[0] + histogramOfLevelTwo[1] + histogramOfLevelTwo[4] + histogramOfLevelTwo[5]
    histogramOfLevelOne[1] = histogramOfLevelTwo[2] + histogramOfLevelTwo[3] + histogramOfLevelTwo[6] + histogramOfLevelTwo[7]
    histogramOfLevelOne[2] = histogramOfLevelTwo[8] + histogramOfLevelTwo[9] + histogramOfLevelTwo[12] + histogramOfLevelTwo[13]
    histogramOfLevelOne[3] = histogramOfLevelTwo[10] + histogramOfLevelTwo[11] + histogramOfLevelTwo[14] + histogramOfLevelTwo[15]

    # level 0
    histogramOfLevelZero = histogramOfLevelOne[0] + histogramOfLevelOne[1] + histogramOfLevelOne[2] + histogramOfLevelOne[3]

    if level == 0:
        return histogramOfLevelZero
    elif level == 1:
        tempZero = histogramOfLevelZero.flatten() * 0.5
        tempOne = histogramOfLevelOne.flatten() * 0.5
        result = np.concatenate((tempZero, tempOne))
        return result
    elif level == 2:
        tempZero = histogramOfLevelZero.flatten() * 0.25
        tempOne = histogramOfLevelOne.flatten() * 0.25
        tempTwo = histogramOfLevelTwo.flatten() * 0.5
        result = np.concatenate((tempZero, tempOne, tempTwo))
        return result
    else:
        return None

def get_sift_lowe(img):
    features_fname = img + '.sift'
    if os.path.isfile(features_fname) == False:
        is_size_zero = sift.process_image(img, features_fname)
        if is_size_zero:
            os.remove(features_fname)
            sift.process_image(img, features_fname)
    if os.path.isfile(features_fname) and os.path.getsize(features_fname) == 0:
        os.remove(features_fname)
        sift.process_image(img, features_fname)
    locs, desc = sift.read_features_from_file(features_fname)
    return desc

def get_sift(img, with_kp=False):
    raw = cv2.imread(img)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    #descriptor = cv2.SURF()
    descriptor = cv2.SIFT()
    """descriptor = cv2.SIFT(nfeatures=1000,
                    nOctaveLayers=3,
                    contrastThreshold=0.04,
                    edgeThreshold=5)"""
    kp, desc = descriptor.detectAndCompute(gray, None)
    #print img, gray.shape, len(kp), desc.shape

    if with_kp:
        return kp, desc
    else:
        return desc

def get_histogram(k, feature_list, predicted_labels):
    hist = np.zeros(shape = (len(feature_list), k))
    for i, feature in enumerate(feature_list):
        current_hist, bins = np.histogram(feature, bins=k)
        hist[i] = current_hist
    return hist

def removing_null(data, labels):
    new_data = []
    for idx, i in enumerate(data):
        if i != None:
            new_data.append(i)
        else:
            print "Find null : %s" % idx
            del labels[idx]
    return new_data

def get_train_test(TEST):
    FOOD_PATH = "/home/carpedm20/data/food100/"
    SINGLE_FOOD = "/home/carpedm20/data/food100/%s/crop_*.jpg"

    if TEST:
        food1 = glob(SINGLE_FOOD % '1')
        print "\nfood1 : %s" % len(food1)
        food2 = glob(SINGLE_FOOD % '36')
        print "food2 : %s" % len(food2)
        food3 = glob(SINGLE_FOOD % '23')
        print "food3 : %s" % len(food3)
        food4 = glob(SINGLE_FOOD % '17')
        print "food4 : %s" % len(food4)
        food5 = glob(SINGLE_FOOD % '12')
        print "food5 : %s" % len(food5)
        food6 = glob(SINGLE_FOOD % '87')
        print "food6 : %s" % len(food6)
        food7 = glob(SINGLE_FOOD % '19')
        print "food7 : %s" % len(food7)
        food8 = glob(SINGLE_FOOD % '22')
        print "food8 : %s" % len(food8)
        food9 = glob(SINGLE_FOOD % '20')
        print "food9 : %s" % len(food9)
        food10 = glob(SINGLE_FOOD % '16')
        print "food10 : %s" % len(food10)
        foods = food1+food2+food3+food4+food5+food6+food7+food8+food9+food10
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

    #if not TEST:
    if True:
        shuffle(train)
        shuffle(test)

    if TEST:
        pass
        #train = train[:len(train)/50]
        #test = test[:len(test)/50]

    print "\ntrain : %s" % len(train)
    print "test : %s" % len(test)

    train_images = [x[0] for x in train]
    train_labels = [int(x[1]) for x in train]

    test_images = [x[0] for x in test]
    test_labels = [int(x[1]) for x in test]

    return train_images, train_labels, test_images, test_labels

import smtplib

from email.MIMEImage import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEImage import MIMEImage

from config import *

def send_mail(text, content, filename=''):
    global email_username, email_password
    fromaddr = 'hexa.portal@gmail.com'

    recipients = ["carpedm20@gmail.com"]
    toaddrs  = ", ".join(recipients)

    username = email_username
    password = email_password

    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = text
    msgRoot['From'] = fromaddr
    msgRoot['To'] = toaddrs

    msgAlternative = MIMEMultipart('alternative')
    msgRoot.attach(msgAlternative)

    msgText = MIMEText(content, 'html')
    msgAlternative.attach(msgText)

    if filename is not '':
      img = MIMEImage(open(filename,"rb").read(), _subtype="png")
      img.add_header('Content-ID', '<carpedm20>')
      msgRoot.attach(img)
      
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username,password)
    server.sendmail(fromaddr, recipients, msgRoot.as_string())
    server.quit()
    print " - mail sended"
