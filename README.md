# 이미지 유사도 측정하기

<a href="https://github.com/geonlee0325/image_clustering/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg">

## Image Clustering
+ Default (NUM_IMGS_PER_MODEL = 70)
<pre><code>$ python3 make_labels_pred.py</code></pre>
+ K given
<pre><code>$ python3 make_labels_pred.py 'k'</code></pre>
+ K not given (estimate k)
<pre><code>$ python3 make_labels_pred.py -1</code></pre>

## Evaluation
<pre><code>$ python3 evaluation.py</code></pre>
