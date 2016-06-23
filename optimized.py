from feature import *
from pymongo import MongoClient
from bson.binary import Binary as BsonBinary
import pickle
import os
from operator import itemgetter
import time
import sys

imagelocation = "" #Input Image path
indir = "" #Directory Path

client = MongoClient('mongodb://localhost:27017')



db = client.coil #Insert your database in place of coil
col = db.images #Insert your collection in place of images

class Image(object):
    """docstring for Image"""
    def __init__(self, path):
        self.path = path
        img = cv2.imread(self.path,0)
        imgm = preprocess(img)
        segm = segment(imgm)
        self.glfeature = globalfeature(imgm,16)
        self.llfeature = localfeature(segm)
        self.numberofones = self.glfeature.sum(dtype=int)

start_time = time.time()
count = 0

for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        i1 = Image(f)
        count = count+1
        perc = (count/360)  * 100
        sys.stdout.write("\r%d%%" % perc)
        sys.stdout.flush()
        new_posts = [{'path': i1.path,
             'llfeature': BsonBinary(pickle.dumps(i1.llfeature,protocol=2)),
             'glfeature': BsonBinary(pickle.dumps(i1.glfeature,protocol=2)),
             'numberofones' : int(i1.numberofones)}]
        post_id = col.insert(new_posts)
        # print(post_id)
img = Image(imagelocation)
count = 0
maxglosim = 0
maxlocsim = 0
maximum = 0
gridmax=0
vectormax=0

for f in col.find():
    llfeature = pickle.loads(f['llfeature'])
    glfeature = pickle.loads(f['glfeature'])
    count = count+1
    perc = (count/360)  * 100
    sys.stdout.write("\r%d%%" % perc)
    sys.stdout.flush()
    locsim = np.absolute((llfeature-img.llfeature).sum())
    glosim = np.logical_xor(glfeature,img.glfeature).sum()
    distance = locsim+glosim
    if(glosim>maxglosim):
        gridmax = glfeature
        maxglosim=glosim
    if(locsim>maxlocsim):
        maxlocsim=locsim
        vectormax = llfeature
    if(distance>maximum):
        vectmostdif= llfeature
        gridmostdif = glfeature
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

for f in col.find():
    llfeature = pickle.loads(f['llfeature'])
    glfeature = pickle.loads(f['glfeature'])
    count = count+1
    perc = (count/360)  * 100
    sys.stdout.write("\r%d%%" % perc)
    sys.stdout.flush()
    g1 = gloDist(glfeature,img.glfeature)
    l1 = locDist(llfeature,img.llfeature)
    sim = ((2-(g1+l1))/2)*100
    ranking.append([sim,f['path']])

search_time = time.time()
print("\nTotal Searching Time : {0:.2f} seconds".format(search_time-processed_time))
print("\nTotal Time : {0:.2f} seconds".format(search_time-start_time))
ranking = sorted(ranking, key=itemgetter(0),reverse=True)
#Ranking : Results in a list
