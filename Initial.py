from feature import *
from prelib import preprocess
import os
from operator import itemgetter
import time
import sys

import pickle as pickle

imagelocation = "" #Input Image path
indir = "" #Directory Path

class Image(object):
    def __init__(self, path):
        self.path = path
        img = cv2.imread(self.path,0)
        imgm = preprocess(img)
        segm = segment(imgm)
        self.glfeature = globalfeature(imgm,16)
        self.llfeature = localfeature(segm)
        self.numberofones = self.glfeature.sum(dtype=int)

img = Image(imagelocation)

maxglosim = 0
maxlocsim = 0
maximum = 0
count = 0
start_time = time.time()
print("Processing : ")
for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        i1 = Image(f)
        count = count+1
        perc = (count/360)  * 100
        sys.stdout.write("\r%d%%" % perc)
        sys.stdout.flush()
        locsim = np.absolute((i1.llfeature-img.llfeature).sum())
        glosim = np.logical_xor(img.glfeature,i1.glfeature).sum()
        distance = locsim+glosim
        if(glosim>maxglosim):
            gridmax = i1.glfeature
            maxglosim=glosim
        if(locsim>maxlocsim):
            maxlocsim=locsim
            vectormax = i1.llfeature
        if(distance>maximum):
            vectmostdif= i1.llfeature
            gridmostdif = i1.glfeature
            imgmax = i1
            maximum = distance
maxilocsim = np.absolute((vectormax-img.llfeature).sum())

maxiglosim = np.logical_xor(gridmax,img.glfeature).sum()
processed_time = time.time()
print("\nTotal Processing Time : {0:.2f} seconds".format(processed_time-start_time))
def gloDist(gridA,gridB):
    glosim = np.logical_xor(gridA,gridB).sum()
    return glosim/maxiglosim
def locDist(vectorA,vectorB):
    locsim = np.absolute((vectorA-vectorB).sum())
    return locsim/maxilocsim
ranking = []
count = 0
print("\nSearching:")
for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        img1 = Image(f)
        count = count+1
        perc = (count/360)  * 100
        sys.stdout.write("\r%d%%" % perc)
        sys.stdout.flush()
        g1 = gloDist(img1.glfeature,img.glfeature)
        l1 = locDist(img1.llfeature,img.llfeature)
        sim = ((2-(g1+l1))/2)*100
        ranking.append([sim,f])
search_time = time.time()
print("\nTotal Searching Time : {0:.2f} seconds".format(search_time-processed_time))
print("\nTotal Time : {0:.2f} seconds".format(search_time-start_time))
ranking = sorted(ranking, key=itemgetter(0),reverse=True)

#Results stored in ranking
