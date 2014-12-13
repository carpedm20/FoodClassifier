import os
import operator
from pprint import pprint

dict = {}
for i in os.listdir("/home/carpedm20/data/food100"):
    dict[i] = len(os.listdir("/home/carpedm20/data/food100/"+i))

print pprint(sorted(dict.items(), key=operator.itemgetter(1)))
