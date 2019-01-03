import keras
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import adam
from keras.utils import to_categorical
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import metrics
import numpy as np
import os

class_cnt = -1

def vgg19_feature_extraction(dataset_path):
    """
    Extract features using VGG19
    """
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

    # Get features of all images using VGG19
    X = []
    Y = []
    model_cnt = 0
    model_index = {}
    img_list = os.listdir(dataset_path)
    img_list.sort()
    temp_cnt = 0
    for img_file in img_list:
        if temp_cnt % 100 == 0:
            print("VGG19 ", round(temp_cnt/len(img_list)*100,3), "% complete", end='\r')
        temp_cnt = temp_cnt + 1
        img_path = dataset_path + img_file
        img = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        block5_pool_features = model.predict(x).flatten()

        X.append(block5_pool_features) 
        if 'aug' in dataset_path:
            model_id = img_file.split('_')[1]
        else:
            model_id = img_file.split('_')[0]

        if model_id in model_index:
            Y.append(model_index[model_id])
        else:
            model_index[model_id] = model_cnt
            Y.append(model_cnt)
            model_cnt = model_cnt + 1  

    X = np.asarray(X)
    Y_lab = np.asarray(Y)
    Y_cat = to_categorical(Y_lab)

    return X, Y_lab, Y_cat


def random_sampling(X_data, Y_data, data_cnt):
    """
    Randomly sample combinations
    """
    X = []
    Y = []
   
    for i in range(data_cnt):
        same_index = np.argwhere(Y_data==Y_data[i])
        diff_index = np.argwhere(Y_data!=Y_data[i])
        for j in range(200):
            if np.random.rand() < 0.5:
                ind_1 = same_index[np.random.randint(len(same_index))]
                ind_2 = same_index[np.random.randint(len(same_index))]
                X.append(np.multiply(X_data[ind_1],X_data[ind_2]))
                Y.append(1)
            else:
                ind_1 = diff_index[np.random.randint(len(diff_index))]
                ind_2 = diff_index[np.random.randint(len(diff_index))]
                X.append(np.multiply(X_data[ind_1],X_data[ind_2]))
                Y.append(0)
   
    X = np.asarray(X).reshape(data_cnt*200, 256)
    Y = np.asarray(Y)

    return X, Y


def all_combinations(X_data, Y_data):
    """
    Match all combinations
    """
    X = []
    Y = []
    X_index = []
    data_cnt = 0

    for i in range(len(X_data)):
        for j in range(i+1,len(X_data)):
            if np.random.rand() < 0.1:
                data_cnt = data_cnt + 1
                X_index.append([i,j])
                X.append(np.multiply(X_data[i],X_data[j]))
                if Y_data[i] == Y_data[j]:
                    Y.append(1)
                else:
                    Y.append(0)

    X = np.asarray(X).reshape(data_cnt, 256)
    Y = np.asarray(Y)

    return X, Y, X_index


def fully_connected_feature_extraction(X_train, Y_train):
    """
    Extract features using FC layers
    """

    random_normal = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        
    # First Layer (Input Layer)
    model = Sequential()
    model.add(Dense(256, input_dim=512*7*7, init=random_normal))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Second Layer (Hidden Layer 1)
    model.add(Dense(256, init=random_normal))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # Third Layer (Output Layer)
    num_class = class_cnt
    model.add(Dense(num_class, init=random_normal))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
  
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    training_epochs = 20
    batch_size = 256
    model.fit(X_train, Y_train, nb_epoch=training_epochs, batch_size=batch_size)

    model_json = model.to_json()
    with open("fc_nn_model.json","w") as json_file:
        json_file.write(model_json)
    model.save_weights("fc_nn_weight.h5")

    feature_model = Sequential()
    feature_model.add(Dense(256, input_dim=512*7*7, weights=model.layers[0].get_weights()))
    feature_model.add(BatchNormalization())
    
    feature_train = feature_model.predict(X_train)
    return feature_train


def classifer(X_train, Y_train, X_valid, Y_valid):
    """
    Classify whether two data are in the same cluster
    """

    random_normal = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)

    # First Layer (Input Layer)
    model = Sequential()
    model.add(Dense(512, input_dim=256, init=random_normal))
    model.add(Dropout(0.35))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Second Layer (Hidden Layer 1)
    model.add(Dense(256, init=random_normal))
    model.add(Dropout(0.35))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Third Layer (Hidden Layer 2)
    model.add(Dense(128, init=random_normal))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Fourth Layer (Output Layer)
    model.add(Dense(1, init=random_normal))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    training_epochs = 10
    batch_size = 256
    model.fit(X_train, Y_train, nb_epoch=training_epochs, batch_size=batch_size)

    model_json = model.to_json()
    with open("classifier_model.json","w") as json_file:
        json_file.write(model_json)
    model.save_weights("classifier_weight.h5")



if __name__ == '__main__':
    
    # VGG19 FEATURE EXTRACTION
    X_train, Y_train_lab, Y_train_cat = vgg19_feature_extraction("../train_augmentation/")
    class_cnt = len(set(Y_train_lab))

    # FC FEATURE EXTRACTION
    X_train = fully_connected_feature_extraction(X_train, Y_train_cat)

    # RANDOM SAMPLING
    X_train_sample, Y_train_sample = random_sampling(X_train, Y_train_lab, class_cnt)

    # CLASSIFIER NN
    classifer(X_train_sample, Y_train_sample, X_train_sample, Y_train_sample)
