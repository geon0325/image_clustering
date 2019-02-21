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

if __name__ == '__main__':
    
    # VGG19 FEATURE EXTRACTION
    X_test, Y_test_lab, Y_test_cat = vgg19_feature_extraction(IMG_DIR + '/')
    
    # FC FEATURE EXTRACTION
    X_feature = fc_feature_extraction(X_test)
    X_feature.tofile('./features.npy')
    Y_test_lab.tofile('./labels.npy')
