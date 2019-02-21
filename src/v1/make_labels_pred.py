from training import * 
from config import *
from keras.models import model_from_json
import sys

def fc_feature_extraction(X_test):
    json_file = open("fc_nn_model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    fc_model = model_from_json(loaded_model_json)
    fc_model.load_weights("fc_nn_weight.h5")

    feature_model = Sequential()
    feature_model.add(Dense(256, input_dim=512*7*7, weights=fc_model.layers[0].get_weights()))
    feature_model.add(BatchNormalization())

    feature = feature_model.predict(X_test)
    return feature


def test_data_combination(X_feature, drop_prob=0.0):
    X = []
    X_index = []
    data_cnt = 0
    for i in range(len(X_feature)):
        for j in range(i+1, len(X_feature)):
            if np.random.rand() <= (1.0 - drop_prob):
                data_cnt = data_cnt + 1
                X_index.append([i,j])
                X.append(np.multiply(X_feature[i], X_feature[j]))
    
    X = np.asarray(X).reshape(data_cnt, 256)
    return X, X_index


def classify_combination(X_comb):
    json_file = open("classifier_model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    classifier_model = model_from_json(loaded_model_json)
    classifier_model.load_weights("classifier_weight.h5")

    pred = classifier_model.predict(X_comb)
    return pred


def objective_value(pred_link, pred_label, X_index):
    J = 0
    for i in range(len(pred_link)):
        data_ind1 = X_index[i][0]
        data_ind2 = X_index[i][1]
        if pred_label[data_ind1] == pred_label[data_ind2]:
            if pred_link[i] < 0.5:
                J = J + (1.0 / pred_link[i])
        else:
            if pred_link[i] >= 0.5:
                J = J + (1.0 / (1.0 - pred_link[i]))
    return J


def clustering_k_not_given(X_feature, pred_link, X_index):
    J_list = []
    for k in range(2,int(min(MAX_CLUSTER,len(X_feature)/2 + 1))):
        kmeans = KMeans(n_clusters=k, n_init=20, n_jobs=4, init="k-means++")
        pred_label = kmeans.fit_predict(X_feature)
        J = objective_value(pred_link, pred_label, X_index)
        J_list.append(J)
    
    optimal_k = -1
    try:
        window_size = 5
        J_sum_min = 1e10
        for i in range(len(J_list) - window_size + 1):
            if sum(J_list[i:i+window_size]) < J_sum_min:
                J_sum_min = sum(J_list[i:i+window_size])
                optimal_k = int((2 + i + 2 + i + window_size) / 2)
    except:
        optimal_k = J_list.index(min(J_list)) + 2
    if optimal_k == -1:
        optimal_k = J_list.index(min(J_list)) + 2
        
    kmeans = KMeans(n_clusters=optimal_k, n_init=20, n_jobs=4, init="k-means++")
    pred_label = kmeans.fit_predict(X_feature)
    return pred_label


def clustering_k_given(X_feature, k):
    kmeans = KMeans(n_clusters=k, n_init=20, n_jobs=4, init="k-means++")
    pred_label = kmeans.fit_predict(X_feature)
    return pred_label


def clustering_default(X_feature):
    k = int(len(X_feature) / NUM_IMGS_PER_MODEL)
    if k > len(X_feature) or k < 2:
        X_comb, X_index = test_data_combination(X_feature, DROP_P)
        pred_link = classify_combination(X_comb)
        pred_label = clustering_k_not_given(X_feature, pred_link, X_index)
    else:
        kmeans = KMeans(n_clusters=k, n_init=20, n_jobs=4, init="k-means++")
        pred_label = kmeans.fit_predict(X_feature)
    return pred_label


if __name__ == '__main__':
    
    # VGG19 FEATURE EXTRACTION
    X_test, Y_test_lab, Y_test_cat = vgg19_feature_extraction('../../valid_dataset/')
    
    # FC FEATURE EXTRACTION
    X_feature = fc_feature_extraction(X_test)
    print(X_feature.shape)
    X_feature.tofile('./features.npy')
    print(Y_test_cat.shape)
    print(Y_test_lab.shape)
    print(Y_test_lab.dtype)
    Y_test_lab.tofile('./labels.npy')
    # # CLUSTERING
    # if len(sys.argv) == 1:
    #     # default k
    #     pred_label = clustering_default(X_feature)
    #
    # elif len(sys.argv) > 1:
    #     if int(sys.argv[1]) == -1:
    #         # k not given (estimate k)
    #         # DATA COMBINATION
    #         if len(sys.argv) == 2:
    #             X_comb, X_index = test_data_combination(X_feature, DROP_P)
    #         else:
    #             X_comb, X_index = test_data_combination(X_feature, float(sys.argv[2]))
    #         # CLASSIFIER NN
    #         pred_link = classify_combination(X_comb)
    #         # CLUSTERING
    #         pred_label = clustering_k_not_given(X_feature, pred_link, X_index)
    #
    #     else:
    #         # k given
    #         k = int(sys.argv[1])
    #         pred_label = clustering_k_given(X_feature, k)
    #
    # # FILE WRITE
    # with open(os.path.join(DATA_DIR, LABELS_PRED + ".txt"), 'w') as f:
    #     f.writelines([str(line) + "\n" for line in pred_label])
