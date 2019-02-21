# 이미지 유사도 측정하기

<a href="https://github.com/geonlee0325/image_clustering/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg">

## Model Architecture
1. Preprocessing
  - 중복 이미지 제거
  - Image Augmentation
2. Feature Extraction
  - VGGNet
  - Fully Connected NN
3. Clustering
  - K-Means
  - Estimating K

## Image Clustering (Test)
NEW VERSION
+ Default (NUM_IMGS_PER_MODEL = 70)
<pre><code>$ bash make_pred.sh</code></pre>
OLD VERSION
+ Default (NUM_IMGS_PER_MODEL = 70)
<pre><code>$ python3 make_labels_pred.py</code></pre>
+ k (# of clusters) given
<pre><code>$ python3 make_labels_pred.py 'k'</code></pre>
+ k (# of clusters) not given (estimate k)
<pre><code>$ python3 make_labels_pred.py -1</code></pre>

Running time depends on the size of the testing dataset.

## Evaluation
<pre><code>$ python3 make_labels_true.py
$ python3 evaluation.py</code></pre>

## Training
Our repository contains pre-trained model of two classifiers.
+ fc_nn_model.json
+ fc_nn_weight.h5

These two files are each feature extracting FC layer's structure and weights. 
+ classifier_model.json
+ classifier_weight.h5

If you want to retrain the model, you can run "training.py" with training dataset path following.
<pre><code>$ python3 training.py ../train_dataset</pre></code>

## Notes
Our model contains additional configuration "DROP_P" in config.py. It enables our model run using less memory. If your computer's memory seems to be unbearable, raise DROP_P by modifying values in config.py line 22
<pre><code>DROP_P = 0.0</code></pre>
or run make_labels_pred.py giving additional argument.
<pre><code>$ python3 make_labels_pred.py -1 0.95</code></pre>

