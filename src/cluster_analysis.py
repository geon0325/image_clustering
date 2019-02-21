import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from scipy import spatial
import math
from operator import itemgetter
from glob import glob
from config import *
#import delf_make_ranking

def evaluation(true_label, est_label):
    return metrics.adjusted_rand_score(true_label, est_label)

def k_means(feature, k):
    clustering = KMeans(n_clusters=k, random_state=0, init='k-means++').fit(feature)
    #return clustering.cluster_centers_
    return clustering.labels_, clustering.cluster_centers_

def distance(x_1, x_2):
    l = len(x_1)
    d = 0
    for i in range(l):
        d = d + math.pow(x_1[i]-x_2[i],2)
    return d

data_size = len(glob(IMG_DIR + '/*'))
n_clusters = int(data_size / NUM_IMGS_PER_MODEL)
feature_dimension = 256

label = np.fromfile('labels.npy', dtype=np.int64)
#label = np.reshape(label, (data_size, 1))
feature = np.fromfile('features.npy', dtype=np.float32)
feature = np.reshape(feature, (data_size, feature_dimension))

print(label.shape)
print(feature.shape)

f = open('kmeans_centroids.txt','w')
f_2 = open('label_feature.txt', 'w')
est, centroid = k_means(feature, n_clusters)

kmeans_result = [[] for _ in range(n_clusters)]

ind_cnt = 0
for e in est:
    kmeans_result[e].append(ind_cnt)
    ind_cnt = ind_cnt + 1

dist_result = [[] for _ in range(n_clusters)]
for i in range(n_clusters):
    dist = []
    for j in kmeans_result[i]:
        dist.append([j,float(distance(feature[j],centroid[i]))])
    sorted(dist, key=lambda l:l[1])
    dist.sort(key=itemgetter(1), reverse=False)
    print(dist)
    for d in dist:
        dist_result[i].append(d[0])
    

data = ''
for i in range(n_clusters):
    for j in dist_result[i]:
        data = data + str(j) + ','
    data = data + '\n'
f.write(data)

