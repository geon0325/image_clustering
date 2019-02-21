#!/bin/bash

#extract feature use pretrianed model
python extract_features.py

#Do Kmeans clustering for find centroids
python cluster_analysis.py

#make label prediction using DELF (DEep Local Features)
python make_labels_pred.py
