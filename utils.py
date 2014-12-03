import numpy as np
import cv2
import cPickle

from glob import glob
from random import shuffle
from time import gmtime, strftime

from skimage.io import imread
from sklearn.cross_validation import train_test_split

def save_pickle(name, obj):
    file_name = "%s_%s.pkl" % (name, strftime("%m%d-%H%M", gmtime()))
    with open(file_name, "wb") as f:
        cPickle.dump(obj, f)

def get_color_histogram(img):
    raw = cv2.imread(img)
    hists = []
    color = ('b','g','r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([raw],[i],None,[256],[0,255])
        cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
        hists.append(np.int32(np.around(hist)).reshape((len(hist),)))
    return np.concatenate(hists, axis = 0)

def get_sift(img):
    raw = cv2.imread(img)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    #sift = cv2.SURF()
    sift = cv2.SIFT()
    """sift = cv2.SIFT(nfeatures=1000,
                    nOctaveLayers=3,
                    contrastThreshold=0.04,
                    edgeThreshold=5)"""
    kp, desc = sift.detectAndCompute(gray, None)
    #print img, gray.shape, len(kp), desc.shape
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

    if not TEST:
        shuffle(train)
        shuffle(test)

    if TEST:
        train = train[:len(train)/10]
        test = test[:len(test)/10]

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
