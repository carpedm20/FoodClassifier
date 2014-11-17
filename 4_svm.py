from random import shuffle

from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.cluster import KMeans
from glob import glob

TEST = True
FOOD_PATH = "/home/carpedm20/data/food100/"

if TEST:
    food1 = glob("/home/carpedm20/data/food100/1/*.jpg")
    print "\nfood1 : %s" % len(food1)
    food2 = glob("/home/carpedm20/data/food100/36/*.jpg")
    print "food2 : %s" % len(food2)
    food3 = glob("/home/carpedm20/data/food100/23/*.jpg")
    print "food3 : %s" % len(food3)
    foods = food1 + food2 + food3
else:
    foods = glob.glob("/home/carpedm20/data/food100/*/*.jpg")

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

print "\ntrain : %s" % len(train)
print "test : %s" % len(test)

train_labels = [int(x[1]) for x in train]
test_labels = [int(x[1]) for x in test]


