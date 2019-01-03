# 이미지 유사도 측정하기

<a href="https://github.com/geonlee0325/image_clustering/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg">

## Image Clustering (Test)
+ Default (NUM_IMGS_PER_MODEL = 70)
<pre><code>$ python3 make_labels_pred.py</code></pre>
+ k (# of clusters) given
<pre><code>$ python3 make_labels_pred.py 'k'</code></pre>
+ k (# of clusters) not given (estimate k)
<pre><code>$ python3 make_labels_pred.py -1</code></pre>

## Evaluation
<pre><code>$ python3 make_labels_true.py
$ python3 evaluation.py</code></pre>

## Training
Our repository contains pre-trained model of two classifiers.
+ fc_nn_model.json
+ fc_nn_weight.h5
This two files are each feature extracting FC layer's structure and weights. 
+classifier_model.json
+classifier_weight.h5
This two files are each pair classifier structure and weights.

If you want to retrain our model, you can run "training.py"
<pre><code>$ python3 training.py</pre></code>
This code retrain our model and save above 4 files in directory that training.py is in.

## Notes
Our model contains additional configuration "DROP_P" in config.py
that means drop probability of feature pairs.
It makes our model can run using less memory.

In config.py line 22
<pre><code>DROP_P = 0.0</code></pre>


