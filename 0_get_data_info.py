import os
import operator
from pprint import pprint
from glob import glob

dict = {}
for i in os.listdir("/home/carpedm20/data/food100"):
    dict[i] = len(os.listdir("/home/carpedm20/data/food100/"+i))

print pprint(sorted(dict.items(), key=operator.itemgetter(1)))

dict_original = {}

for i in os.listdir("/home/carpedm20/data/food100"):
    ll=glob("/home/carpedm20/data/food100/"+i+"/*.jpg")
    for j in ll:
        if any(word in j for word in ["crop","rotate","flip"]):
            pass
        else:
            try:
                dict_original[i].append(j)
            except:
                dict_original[i] = []
                dict_original[i].append(j)

dict_original_len = {}
for i, j in dict_original.items():
    dict_original_len[i] = len(j)
#print pprint(sorted(dict_original_len.items(), key=operator.itemgetter(1)))
d = sorted(dict_original_len.items(), key=operator.itemgetter(1))
for j in d:
    print j[1]
