import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf
import tensorflow_hub as hub
from glob import glob
from config import *

import random
import time

class delf_wrapper():

    '''
    based on "https://github.com/tensorflow/hub/blob/42d7bcea698b614dbc20cee52091108c0753d990/examples/colab/tf_hub_delf_module.ipynb"
    '''


    def __init__(self, image_folder_path, output_path):
        self.image_folder_path = image_folder_path
        self.output_path = output_path
        self.results_dict = {}

    def image_input_fn(self, image_path):
        '''
        read image from image path
        :return: image tensor
        '''
        value = tf.read_file(image_path)
        image_tf = tf.image.decode_jpeg(value, channels=3)
        return tf.image.convert_image_dtype(image_tf, tf.float32)

    def input_pipeline(self):
        '''
        define input pipeline
        :return: iterator
        '''
        image_path = sorted(glob(self.image_folder_path + '/*.jpg'))
        self.num_image = len(image_path)

        dataset = tf.data.Dataset.from_tensor_slices(image_path)
        dataset = dataset.map(self.image_input_fn)
        dataset = dataset.prefetch(10)

        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()

    def init_model(self):
        '''
        initialize delf model
        '''
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.FATAL)

        self.model = hub.Module('https://tfhub.dev/google/delf/1')

        self.input_ph = tf.placeholder(tf.float32, shape=[None, None, 3], name="input_image")

        next_image = self.input_pipeline()
        module_inputs = {
            'image': next_image,
            'score_threshold': 100.0,
            'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
            'max_feature_num': 1000,
        }
        module_inputs_ph = {
            'image': self.input_ph,
            'score_threshold': 100.0,
            'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
            'max_feature_num': 1000,
        }

        self.output = self.model(module_inputs, as_dict=True)
        self.output_ph = self.model(module_inputs_ph, as_dict=True)

    def extract_feature_path(self, image_path, image):
        '''
        extract features from image path and image
        :return: results_dict with location and descriptors of image
        '''
        # maybe have to change monitoredsession later?
        with tf.train.MonitoredSession() as sess:
            features = sess.run([self.output_ph['locations'],
                                self.output_ph['descriptors']], feed_dict={self.input_ph: image})
            self.results_dict[image_path] = features
            return self.results_dict

    def extract_feature(self):
        '''
        extract features from input image pipeline
        :return: results_dict with location and descriptors of image
        '''

        image_paths = sorted(glob(self.image_folder_path + '/*.jpg'))
        image_len = len(image_paths)
        cnt = 0
        # maybe have to change monitoredsession later?

        with tf.train.MonitoredSession() as sess:
            print('start extracting local features')
            for image_path in image_paths:
                cnt += 1
                if cnt % 100 == 0:
                    print('{} / {}'.format(cnt, image_len))
                self.results_dict[image_path] = sess.run([self.output['locations'], self.output['descriptors']])
        return self.results_dict

    def assign_cluster(self, centroid_idx):
        image_path = sorted(glob(self.image_folder_path + '/*.jpg'))
        print(image_path)
        center_image_path = [image_path[i] for i in centroid_idx]
        p_len = len(image_path)

        cKDtrees = []
        location_all = []
        descriptor_all = []
        for p in center_image_path:
            locations, descriptors = self.results_dict[p]
            descriptor_all.append(descriptors)
            cKDtrees.append(cKDTree(descriptors))
            location_all.append(locations)

        # test
        #cKDtrees, location_all = self.reduce_center(cKDtrees, location_all, descriptor_all)

        print(len(cKDtrees), len(location_all))
        cluster = []
        for i in range(p_len):
            if i % 10 == 0:
                print('{} / {}'.format(i, p_len))
            cur_idx = self.match_images(image_path[i], cKDtrees, location_all)
            cluster.append(cur_idx)
        with open(self.output_path + 'labels_pred.txt', 'w') as f:
            for idx in cluster:
                f.write(str(idx) + '\n')

    def match_image(self, kdtrees, location_all, locations, tree_idx, descriptors, distance_threshold=0.8):
        num_features = locations.shape[0]
        # Find nearest-neighbor matches using a KD tree.
        _, indices = kdtrees[tree_idx].query(
            descriptors, distance_upper_bound=distance_threshold)

        num_features_tmp = location_all[tree_idx].shape[0]
        # Select feature locations for putative matches.
        locations_1_to_use = np.array([
                                          locations[i,]
                                          for i in range(num_features)
                                          if indices[i] != num_features_tmp
                                          ])
        locations_tmp_to_use = np.array([
                                            location_all[tree_idx][indices[i],]
                                            for i in range(num_features)
                                            if indices[i] != num_features_tmp
                                            ])

        # Perform geometric verification using RANSAC.
        # handling exception
        if locations_1_to_use.shape == (0,) or locations_tmp_to_use.shape == (0,):
            return 0
        try:
            _, inliers = ransac(
                (locations_1_to_use, locations_tmp_to_use),
                AffineTransform,
                min_samples=3,
                residual_threshold=20,
                max_trials=50)
        except:
            return 0

        if inliers is not None and inliers is not []:
            return sum(inliers)
        else:
            return 0

    def reduce_center(self, cKDtrees, location_all, descriptor_all):
        len_center_fix = len(cKDtrees)

        len_center = len_center_fix
        # print(len_center_fix)
        while len_center > int(len_center_fix / 3):
            sample_idx = random.sample(list(np.arange(0, len_center)), 7)

            # print(sample_idx)

            samples = []

            for j in sample_idx:
                num_inliers = []
                for i in range(len_center):
                    num_inliers.append(self.match_image(cKDtrees, location_all, location_all[j], i, descriptor_all[j]))
                    # print(num_inliers)
                samples.append(num_inliers)

            # print(samples)
            max_idx = sample_idx[np.argmax(np.max(samples, 1), 0)]

            # print(max_idx)
            len_center = len_center - 1
            cKDtrees = np.delete(cKDtrees, max_idx)
            location_all = np.delete(location_all, max_idx)
            descriptor_all = np.delete(descriptor_all, max_idx)

        return cKDtrees, location_all

    def match_images(self, path, kdtrees, location_all):
        '''
        matching two image
        :return: the number of matching points
        '''
        distance_threshold = 0.8

        # Read features.
        locations, descriptors = self.results_dict[path]
        num_features = locations.shape[0]
        len_t = len(location_all)

        num_inlier = []
        for tree_idx in range(len_t):
            num_inlier.append(self.match_image(kdtrees, location_all, locations, tree_idx, descriptors))
        return np.argmax(num_inlier)


np.seterr(divide='ignore', invalid='ignore')

with open('kmeans_centroids.txt', 'r') as f:
    indices = f.readlines()
    centroids = []
    for idx in indices:
        centroids.append(int(idx.split(',')[0]))

image_paths = sorted(glob(IMG_DIR + '/*.jpg'))
print(centroids)

model = delf_wrapper(IMG_DIR, DATA_DIR)
model.init_model()
model.extract_feature()
model.assign_cluster(centroids)
