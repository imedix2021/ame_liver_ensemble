from __future__ import print_function
import keras
import sys
print(sys.version)
print(sys.path)
import struct
import numpy as np
import os, cv2, zipfile, io, re, glob
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve
import PIL.Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from keras.models import model_from_json
from keras.models import Model, load_model
from keras.layers.core import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns

import pandas as pd

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from efficientnet.keras import EfficientNetB0
from efficientnet.keras import EfficientNetB1
from efficientnet.keras import EfficientNetB2
from efficientnet.keras import EfficientNetB3
from efficientnet.keras import EfficientNetB4
from efficientnet.keras import EfficientNetB5
from efficientnet.keras import EfficientNetB6
import shutil

import numpy as np 
from skimage.io import imread
from IPython.display import Image
#from efficientnet.keras import EfficientNetB3 as Net
#from efficientnet.keras import center_crop_and_resize, preprocess_input
from tensorflow.keras.preprocessing import image



import gc

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from math import sqrt
from scipy.special import ndtri

from sklearn.metrics import auc

from catboost import CatBoost
from catboost import Pool
from catboost import CatBoostClassifier
import sklearn.metrics
import optuna


# Now build your graph and train it
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import random
#任意のseed値を設定
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

#クラス取得
dirname = os.getcwd()
img_dirs = os.listdir(dirname + "/hikakudata/Test")
img_dirs.sort()
print(img_dirs)
classes = img_dirs
#classes = ['cyst','hcc','hemangioma','meta']

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    seed = 42

    num_classes = len(classes)

    del img_dirs

    # 画像サイズ
    image_size = 256
    image_size2 = 300

    # 画像を取得し、配列に変換
    def im2array(path):
        X = []
        y = []
        class_num = 0

        for class_name in classes:
            if class_num == num_classes : break
            imgfiles = glob.glob(path + class_name + "/*.bmp")
            for imgfile in imgfiles:
                # 画像読み込み
                image = PIL.Image.open(imgfile)
                # RGB変換
                image = image.convert('RGB')
                # リサイズ
                image = image.resize((image_size, image_size))
                # 画像から配列に変換
                data = np.asarray(image)
                X.append(data)
                y.append(classes.index(class_name))
            class_num += 1

        X = np.array(X)
        y = np.array(y)

        return X, y
    ##testデータ取得
    X_test, y_test = im2array("hikakudata/Test/")

    ## testデータ型の変換
    X_test = X_test.astype('float32')

    # test正規化
    X_test /= 255

    ## test one-hot 変換
    y_test_old = y_test
    y_test = to_categorical(y_test, num_classes = num_classes)

    #trainデータ取得
    X_train, y_train = im2array("hikakudata/Train/")

    X_train = X_train.astype('float32')

    # train正規化
    X_train /= 255
    '''
    mod_X_train =X_train[0][0][:,0]
    print(mod_X_train.shape)
    print(mod_X_train)
    print(X_train)
    print(X_train.shape)
    quit()
    print(y_train)
    print(y_train.shape)
    print(pd.DataFrame(y_train))
    quit()
    '''

    # train one-hot 変換
    y_train_old = y_train
    y_train = to_categorical(y_train, num_classes = num_classes)

    # 学習データを、学習用と検証用に分ける


    #df_X_train  = pd.DataFrame(X_train)
    #df_X_test  = pd.DataFrame(X_test)
    
    #df_X_train  = X_train
    #df_X_test  = X_test
    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train_old  shape:", y_train_old.shape)
    print("pd_DataFrame_y_train shape:", pd.DataFrame(y_train))

    models = []
    model_dir = './model/'
    weights_dir = './weights/'
    
    ##Top1_ResNeXt101_all
    f1 = open(model_dir +"Liver_256_noaugResNeXt101_all_20220423_rsf_batch32.json", 'r')
    loaded_model_json1 = f1.read()
    f1.close()
    model1 = model_from_json(loaded_model_json1)
    model1.load_weights(weights_dir + 'Liver_256_noaugResNeXt101_all_20220423_rsf_batch32.h5')
    #y_pred1 = model1.predict(X_train, verbose=1)
    #df_X1 = pd.DataFrame(y_pred1)
    model1_X_train_prob = model1.predict(X_train, verbose=1)
    model1_X_test_prob = model1.predict(X_test, verbose=1)
    
    del f1
    del model1
    #del y_pred1
    gc.collect()
    print("Top1: 取り込み完了")
    
    ##Top2_Xception_108
    f2 = open(model_dir +"Liver_256_noaugXception_20210726_rsf.json", 'r')
    loaded_model_json2 = f2.read()
    f2.close()
    model2 = model_from_json(loaded_model_json2)
    model2.load_weights(weights_dir + 'Liver_256_noaugXception_20210726_rsf.h5')
    #y_pred2 = model2.predict(X_train, verbose=1)
    #df_X2 = pd.DataFrame(y_pred2)
    model2_X_train_prob = model2.predict(X_train, verbose=1)
    model2_X_test_prob = model2.predict(X_test, verbose=1)
    del f2
    del model2
    #del y_pred2
    gc.collect()
    print("Top2: 取り込み完了")
     
    ##Top3_InceptionResNetV2_108
    f3 = open(model_dir +"Amed_Liver_256_108_20210726InceptionResNetV2_rsf.json", 'r')
    loaded_model_json3 = f3.read()
    f3.close()
    model3 = model_from_json(loaded_model_json3)
    model3.load_weights(weights_dir + 'Amed_Liver_256_108_20210726InceptionResNetV2_rsf.h5')
    #y_pred3 = model3.predict(X_train, verbose=1)
    #df_X3 = pd.DataFrame(y_pred3)
    model3_X_train_prob = model3.predict(X_train, verbose=1)
    model3_X_test_prob = model3.predict(X_test, verbose=1)
    del f3
    del model3
    #del y_pred3
    gc.collect()
    print("Top3: 取り込み完了")
    
    #Top4_SeResNeXt50_all
    f4 = open(model_dir +"liver_256_noaugSeResNeXt50_all_20210729_rsf.json", 'r')
    loaded_model_json4 = f4.read()
    f4.close()
    model4 = model_from_json(loaded_model_json4)
    model4.load_weights(weights_dir + 'liver_256_noaugSeResNeXt50_all_20210729_rsf.h5')
    model4_X_train_prob = model4.predict(X_train, verbose=1)
    model4_X_test_prob = model4.predict(X_test, verbose=1)
    del f4
    del model4
    gc.collect()
    print("Top4 : 取り込み完了")

    ##Top5_
    f5 = open(model_dir +"liver_256_noaugResNeXt50_all_20210728_rsf.json", 'r')
    loaded_model_json5 = f5.read()
    f5.close()
    model5 = model_from_json(loaded_model_json5)
    model5.load_weights(weights_dir + 'liver_256_noaugResNeXt50_all_20210728_rsf.h5')
    #y_pred5 = model5.predict(X_train, verbose=1)
    #df_X5 = pd.DataFrame(y_pred5)
    model5_X_train_prob = model5.predict(X_train, verbose=1)
    model5_X_test_prob = model5.predict(X_test, verbose=1)
    del f5
    del model5
    #del y_pred5
    gc.collect()
    print("Top5 : 取り込み完了")
    
    ##Top6_efficientnetB0
    f6 = open(model_dir +"Amed_Liver_-20_size256_20220312efficientnetB0_rsf_batch32_noisy.json", 'r')
    loaded_model_json6 = f6.read()
    f6.close()
    model6 = model_from_json(loaded_model_json6)
    model6.load_weights(weights_dir + 'Amed_Liver_-20_size256_20220312efficientnetB0_rsf_batch32_noisy.h5')
    #y_pred4 = model4.predict(X_train, verbose=1)
    #df_X4 = pd.DataFrame(y_pred4)
    model6_X_train_prob = model6.predict(X_train, verbose=1)
    model6_X_test_prob = model6.predict(X_test, verbose=1)
    del f6
    del model6
    #del y_pred6
    gc.collect()
    print("Top6 : 取り込み完了")
    
    #Top7_SeResNeXt101_all
    f7 = open(model_dir +"liver_256_noaugSeResNeXt101_all_20220419_rsf_batch32.json", 'r')
    loaded_model_json7 = f7.read()
    f7.close()
    model7 = model_from_json(loaded_model_json7)
    model7.load_weights(weights_dir + 'liver_256_noaugSeResNeXt101_all_20220419_rsf_batch32.h5')
    model7_X_train_prob = model7.predict(X_train, verbose=1)
    model7_X_test_prob = model7.predict(X_test, verbose=1)
    del f7
    del model7
    #del y_pred7
    gc.collect()
    print("Top7 : 取り込み完了")

    #Top8_ResNet101_26
    f8 = open(model_dir +"liver_size256_noaugResNet101_26_rsf_batch32_20220423.json", 'r')
    loaded_model_json8 = f8.read()
    f8.close()
    model8 = model_from_json(loaded_model_json8)
    model8.load_weights(weights_dir + 'liver_size256_noaugResNet101_26_rsf_batch32_20220423.h5')
    model8_X_train_prob = model8.predict(X_train, verbose=1)
    model8_X_test_prob = model8.predict(X_test, verbose=1)
    del f8
    del model8
    #del y_pred8
    gc.collect()
    print("Top8 : 取り込み完了")

    #Top9_ResNet50_all
    f9 = open(model_dir +"liver_noaugResNet50_20210727_rsf_batch32_size256.json", 'r')
    loaded_model_json9 = f9.read()
    f9.close()
    model9 = model_from_json(loaded_model_json9)
    model9.load_weights(weights_dir + 'liver_noaugResNet50_20210727_rsf_batch32_size256.h5')
    model9_X_train_prob = model9.predict(X_train, verbose=1)
    model9_X_test_prob = model9.predict(X_test, verbose=1)
    del f9
    del model9
    gc.collect()
    print("Top9 : 取り込み完了")

    #Top10_InceptionV3_249
    f10 = open(model_dir +"Amed_Liver_249_20210726InceptionV3_rsf_batch32_size256.json", 'r')
    loaded_model_json10 = f10.read()
    f10.close()
    model10 = model_from_json(loaded_model_json10)
    model10.load_weights(weights_dir + 'Amed_Liver_249_20210726InceptionV3_rsf_batch32_size256.h5')
    model10_X_train_prob = model10.predict(X_train, verbose=1)
    model10_X_test_prob = model10.predict(X_test, verbose=1)
    del f10
    del model10
    gc.collect()
    print("Top10 : 取り込み完了")


    #Top11_EfficientnetB2_NoisyStudent
    f11 = open(model_dir +"Amed_Liver_-20_256_20220312efficientnetB2_rsf_batch32_noisy.json", 'r')
    loaded_model_json11 = f11.read()
    f11.close()
    model11 = model_from_json(loaded_model_json11)
    model11.load_weights(weights_dir + 'Amed_Liver_-20_256_20220312efficientnetB2_rsf_batch32_noisy.h5')
    model11_X_train_prob = model11.predict(X_train, verbose=1)
    model11_X_test_prob = model11.predict(X_test, verbose=1)
    del f11
    del model11
    gc.collect()
    print("Top11 : 取り込み完了")

    #Top12_EfficientnetB5_NoisyStudent
    f12 = open(model_dir +"Amed_Liver_-20_256_20220423efficientnetB5_rsf_batch32_noisy.json", 'r')
    loaded_model_json12 = f12.read()
    f12.close()
    model12 = model_from_json(loaded_model_json12)
    model12.load_weights(weights_dir + 'Amed_Liver_-20_256_20220423efficientnetB5_rsf_batch32_noisy.h5')
    model12_X_train_prob = model12.predict(X_train, verbose=1)
    model12_X_test_prob = model12.predict(X_test, verbose=1)
    del f12
    del model12
    gc.collect()
    print("Top12 : 取り込み完了")

    #Top13_EfficientnetB3_NoisyStudent
    f13 = open(model_dir +"Amed_Liver_-20_256_20220420efficientnetB3_rsf_batch32_noisy.json", 'r')
    loaded_model_json13 = f13.read()
    f13.close()
    model13 = model_from_json(loaded_model_json13)
    model13.load_weights(weights_dir + 'Amed_Liver_-20_256_20220420efficientnetB3_rsf_batch32_noisy.h5')
    model13_X_train_prob = model13.predict(X_train, verbose=1)
    model13_X_test_prob = model13.predict(X_test, verbose=1)
    del f13
    del model13
    gc.collect()
    print("Top13 : 取り込み完了")

    #Top14_EfficinetnetB6_NoisyStuden
    f14 = open(model_dir +"Amed_Liver_-20_256_20220312efficientnetB6_rsf_batch32_noisy.json", 'r')
    loaded_model_json14 = f14.read()
    f14.close()
    model14 = model_from_json(loaded_model_json14)
    model14.load_weights(weights_dir + 'Amed_Liver_-20_256_20220312efficientnetB6_rsf_batch32_noisy.h5')
    model14_X_train_prob = model14.predict(X_train, verbose=1)
    model14_X_test_prob = model14.predict(X_test, verbose=1)
    del f14
    del model14
    gc.collect()
    print("Top14 : 取り込み完了")

    #Top15_EfficinetnetB1_NoisyStuden
    f15 = open(model_dir +"Amed_Liver_-20_256_20220312efficientnetB1_rsf_noisy.json", 'r')
    loaded_model_json15 = f15.read()
    f15.close()
    model15 = model_from_json(loaded_model_json15)
    model15.load_weights(weights_dir + 'Amed_Liver_-20_256_20220312efficientnetB1_rsf_noisy.h5')
    model15_X_train_prob = model15.predict(X_train, verbose=1)
    model15_X_test_prob = model15.predict(X_test, verbose=1)
    del f15
    del model15
    gc.collect()
    print("Top15 : 取り込み完了")

    #Top16_EfficinetnetB4_NoisyStuden
    f16 = open(model_dir +"Amed_Liver_-20_256_20220312efficientnetB4_rsf_batch32_noisy.json", 'r')
    loaded_model_json16 = f16.read()
    f16.close()
    model16 = model_from_json(loaded_model_json16)
    model16.load_weights(weights_dir + 'Amed_Liver_-20_256_20220312efficientnetB4_rsf_batch32_noisy.h5')
    model16_X_train_prob = model16.predict(X_train, verbose=1)
    model16_X_test_prob = model16.predict(X_test, verbose=1)
    del f16
    del model16
    gc.collect()
    print("Top16 : 取り込み完了")
    
    # 検証データの各クラスの確率
    # 各クラスの確率（5モデル*5seed*4クラス）
    first_probs16 = pd.DataFrame(np.zeros((25440, 16*1*4)))
    for i in range(4):
        first_probs16.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs16.iloc[:, i+4] = model2_X_train_prob[:, i]
        first_probs16.iloc[:, i+8] = model3_X_train_prob[:, i]
        first_probs16.iloc[:, i+12] = model4_X_train_prob[:, i]
        first_probs16.iloc[:, i+16] = model5_X_train_prob[:, i]
        first_probs16.iloc[:, i+20] = model6_X_train_prob[:, i]
        first_probs16.iloc[:, i+24] = model7_X_train_prob[:, i]
        first_probs16.iloc[:, i+28] = model8_X_train_prob[:, i]
        first_probs16.iloc[:, i+32] = model9_X_train_prob[:, i]
        first_probs16.iloc[:, i+36] = model10_X_train_prob[:, i]
        first_probs16.iloc[:, i+40] = model11_X_train_prob[:, i]
        first_probs16.iloc[:, i+44] = model12_X_train_prob[:, i]
        first_probs16.iloc[:, i+48] = model13_X_train_prob[:, i]
        first_probs16.iloc[:, i+52] = model14_X_train_prob[:, i]
        first_probs16.iloc[:, i+56] = model15_X_train_prob[:, i]
        first_probs16.iloc[:, i+60] = model16_X_train_prob[:, i]

        
    test_probs16 = pd.DataFrame(np.zeros((1000, 16*1*4)))
    for i in range(4):
        test_probs16.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs16.iloc[:, i+4] = model2_X_test_prob[:, i]
        test_probs16.iloc[:, i+8] = model3_X_test_prob[:, i]
        test_probs16.iloc[:, i+12] = model4_X_test_prob[:, i]
        test_probs16.iloc[:, i+16] = model5_X_test_prob[:, i]
        test_probs16.iloc[:, i+20] = model6_X_test_prob[:, i]
        test_probs16.iloc[:, i+24] = model7_X_test_prob[:, i]
        test_probs16.iloc[:, i+28] = model8_X_test_prob[:, i]
        test_probs16.iloc[:, i+32] = model9_X_test_prob[:, i]
        test_probs16.iloc[:, i+36] = model10_X_test_prob[:, i]
        test_probs16.iloc[:, i+40] = model11_X_test_prob[:, i]
        test_probs16.iloc[:, i+44] = model12_X_test_prob[:, i]
        test_probs16.iloc[:, i+48] = model13_X_test_prob[:, i]
        test_probs16.iloc[:, i+52] = model14_X_test_prob[:, i]
        test_probs16.iloc[:, i+56] = model15_X_test_prob[:, i]
        test_probs16.iloc[:, i+60] = model16_X_test_prob[:, i]
    
    del model16_X_train_prob
    del model16_X_test_prob
    
    # 書き込み
    np.save('first_probs16.npy', first_probs16)
    np.save('test_probs16.npy', test_probs16)

    first_probs15 = pd.DataFrame(np.zeros((25440, 15*1*4)))
    for i in range(4):
        first_probs15.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs15.iloc[:, i+4] = model2_X_train_prob[:, i]
        first_probs15.iloc[:, i+8] = model3_X_train_prob[:, i]
        first_probs15.iloc[:, i+12] = model4_X_train_prob[:, i]
        first_probs15.iloc[:, i+16] = model5_X_train_prob[:, i]
        first_probs15.iloc[:, i+20] = model6_X_train_prob[:, i]
        first_probs15.iloc[:, i+24] = model7_X_train_prob[:, i]
        first_probs15.iloc[:, i+28] = model8_X_train_prob[:, i]
        first_probs15.iloc[:, i+32] = model9_X_train_prob[:, i]
        first_probs15.iloc[:, i+36] = model10_X_train_prob[:, i]
        first_probs15.iloc[:, i+40] = model11_X_train_prob[:, i]
        first_probs15.iloc[:, i+44] = model12_X_train_prob[:, i]
        first_probs15.iloc[:, i+48] = model13_X_train_prob[:, i]
        first_probs15.iloc[:, i+52] = model14_X_train_prob[:, i]
        first_probs15.iloc[:, i+56] = model15_X_train_prob[:, i]

        
    test_probs15 = pd.DataFrame(np.zeros((1000, 15*1*4)))
    for i in range(4):
        test_probs15.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs15.iloc[:, i+4] = model2_X_test_prob[:, i]
        test_probs15.iloc[:, i+8] = model3_X_test_prob[:, i]
        test_probs15.iloc[:, i+12] = model4_X_test_prob[:, i]
        test_probs15.iloc[:, i+16] = model5_X_test_prob[:, i]
        test_probs15.iloc[:, i+20] = model6_X_test_prob[:, i]
        test_probs15.iloc[:, i+24] = model7_X_test_prob[:, i]
        test_probs15.iloc[:, i+28] = model8_X_test_prob[:, i]
        test_probs15.iloc[:, i+32] = model9_X_test_prob[:, i]
        test_probs15.iloc[:, i+36] = model10_X_test_prob[:, i]
        test_probs15.iloc[:, i+40] = model11_X_test_prob[:, i]
        test_probs15.iloc[:, i+44] = model12_X_test_prob[:, i]
        test_probs15.iloc[:, i+48] = model13_X_test_prob[:, i]
        test_probs15.iloc[:, i+52] = model14_X_test_prob[:, i]
        test_probs15.iloc[:, i+56] = model15_X_test_prob[:, i]

    del model15_X_train_prob
    del model15_X_test_prob

    # 書き込み
    np.save('first_probs15.npy', first_probs15)
    np.save('test_probs15.npy', test_probs15)

    first_probs14 = pd.DataFrame(np.zeros((25440, 14*1*4)))
    for i in range(4):
        first_probs14.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs14.iloc[:, i+4] = model2_X_train_prob[:, i]
        first_probs14.iloc[:, i+8] = model3_X_train_prob[:, i]
        first_probs14.iloc[:, i+12] = model4_X_train_prob[:, i]
        first_probs14.iloc[:, i+16] = model5_X_train_prob[:, i]
        first_probs14.iloc[:, i+20] = model6_X_train_prob[:, i]
        first_probs14.iloc[:, i+24] = model7_X_train_prob[:, i]
        first_probs14.iloc[:, i+28] = model8_X_train_prob[:, i]
        first_probs14.iloc[:, i+32] = model9_X_train_prob[:, i]
        first_probs14.iloc[:, i+36] = model10_X_train_prob[:, i]
        first_probs14.iloc[:, i+40] = model11_X_train_prob[:, i]
        first_probs14.iloc[:, i+44] = model12_X_train_prob[:, i]
        first_probs14.iloc[:, i+48] = model13_X_train_prob[:, i]
        first_probs14.iloc[:, i+52] = model14_X_train_prob[:, i]
        
    test_probs14 = pd.DataFrame(np.zeros((1000, 14*1*4)))
    for i in range(4):
        test_probs14.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs14.iloc[:, i+4] = model2_X_test_prob[:, i]
        test_probs14.iloc[:, i+8] = model3_X_test_prob[:, i]
        test_probs14.iloc[:, i+12] = model4_X_test_prob[:, i]
        test_probs14.iloc[:, i+16] = model5_X_test_prob[:, i]
        test_probs14.iloc[:, i+20] = model6_X_test_prob[:, i]
        test_probs14.iloc[:, i+24] = model7_X_test_prob[:, i]
        test_probs14.iloc[:, i+28] = model8_X_test_prob[:, i]
        test_probs14.iloc[:, i+32] = model9_X_test_prob[:, i]
        test_probs14.iloc[:, i+36] = model10_X_test_prob[:, i]
        test_probs14.iloc[:, i+40] = model11_X_test_prob[:, i]
        test_probs14.iloc[:, i+44] = model12_X_test_prob[:, i]
        test_probs14.iloc[:, i+48] = model13_X_test_prob[:, i]
        test_probs14.iloc[:, i+52] = model14_X_test_prob[:, i]

    del model14_X_train_prob
    del model14_X_test_prob

    # 書き込み
    np.save('first_probs14.npy', first_probs14)
    np.save('test_probs14.npy', test_probs14)

    first_probs13 = pd.DataFrame(np.zeros((25440, 13*1*4)))
    for i in range(4):
        first_probs13.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs13.iloc[:, i+4] = model2_X_train_prob[:, i]
        first_probs13.iloc[:, i+8] = model3_X_train_prob[:, i]
        first_probs13.iloc[:, i+12] = model4_X_train_prob[:, i]
        first_probs13.iloc[:, i+16] = model5_X_train_prob[:, i]
        first_probs13.iloc[:, i+20] = model6_X_train_prob[:, i]
        first_probs13.iloc[:, i+24] = model7_X_train_prob[:, i]
        first_probs13.iloc[:, i+28] = model8_X_train_prob[:, i]
        first_probs13.iloc[:, i+32] = model9_X_train_prob[:, i]
        first_probs13.iloc[:, i+36] = model10_X_train_prob[:, i]
        first_probs13.iloc[:, i+40] = model11_X_train_prob[:, i]
        first_probs13.iloc[:, i+44] = model12_X_train_prob[:, i]
        first_probs13.iloc[:, i+48] = model13_X_train_prob[:, i]

    test_probs13 = pd.DataFrame(np.zeros((1000, 13*1*4)))
    for i in range(4):
        test_probs13.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs13.iloc[:, i+4] = model2_X_test_prob[:, i]
        test_probs13.iloc[:, i+8] = model3_X_test_prob[:, i]
        test_probs13.iloc[:, i+12] = model4_X_test_prob[:, i]
        test_probs13.iloc[:, i+16] = model5_X_test_prob[:, i]
        test_probs13.iloc[:, i+20] = model6_X_test_prob[:, i]
        test_probs13.iloc[:, i+24] = model7_X_test_prob[:, i]
        test_probs13.iloc[:, i+28] = model8_X_test_prob[:, i]
        test_probs13.iloc[:, i+32] = model9_X_test_prob[:, i]
        test_probs13.iloc[:, i+36] = model10_X_test_prob[:, i]
        test_probs13.iloc[:, i+40] = model11_X_test_prob[:, i]
        test_probs13.iloc[:, i+44] = model12_X_test_prob[:, i]
        test_probs13.iloc[:, i+48] = model13_X_test_prob[:, i]

    del model13_X_train_prob
    del model13_X_test_prob

    # 書き込み
    np.save('first_probs13.npy', first_probs13)
    np.save('test_probs13.npy', test_probs13)

    first_probs12 = pd.DataFrame(np.zeros((25440, 12*1*4)))
    for i in range(4):
        first_probs12.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs12.iloc[:, i+4] = model2_X_train_prob[:, i]
        first_probs12.iloc[:, i+8] = model3_X_train_prob[:, i]
        first_probs12.iloc[:, i+12] = model4_X_train_prob[:, i]
        first_probs12.iloc[:, i+16] = model5_X_train_prob[:, i]
        first_probs12.iloc[:, i+20] = model6_X_train_prob[:, i]
        first_probs12.iloc[:, i+24] = model7_X_train_prob[:, i]
        first_probs12.iloc[:, i+28] = model8_X_train_prob[:, i]
        first_probs12.iloc[:, i+32] = model9_X_train_prob[:, i]
        first_probs12.iloc[:, i+36] = model10_X_train_prob[:, i]
        first_probs12.iloc[:, i+40] = model11_X_train_prob[:, i]
        first_probs12.iloc[:, i+44] = model12_X_train_prob[:, i]

    test_probs12 = pd.DataFrame(np.zeros((1000, 12*1*4)))
    for i in range(4):
        test_probs12.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs12.iloc[:, i+4] = model2_X_test_prob[:, i]
        test_probs12.iloc[:, i+8] = model3_X_test_prob[:, i]
        test_probs12.iloc[:, i+12] = model4_X_test_prob[:, i]
        test_probs12.iloc[:, i+16] = model5_X_test_prob[:, i]
        test_probs12.iloc[:, i+20] = model6_X_test_prob[:, i]
        test_probs12.iloc[:, i+24] = model7_X_test_prob[:, i]
        test_probs12.iloc[:, i+28] = model8_X_test_prob[:, i]
        test_probs12.iloc[:, i+32] = model9_X_test_prob[:, i]
        test_probs12.iloc[:, i+36] = model10_X_test_prob[:, i]
        test_probs12.iloc[:, i+40] = model11_X_test_prob[:, i]
        test_probs12.iloc[:, i+44] = model12_X_test_prob[:, i]

    del model12_X_train_prob
    del model12_X_test_prob

    # 書き込み
    np.save('first_probs12.npy', first_probs12)
    np.save('test_probs12.npy', test_probs12)

    first_probs11 = pd.DataFrame(np.zeros((25440, 11*1*4)))
    for i in range(4):
        first_probs11.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs11.iloc[:, i+4] = model2_X_train_prob[:, i]
        first_probs11.iloc[:, i+8] = model3_X_train_prob[:, i]
        first_probs11.iloc[:, i+12] = model4_X_train_prob[:, i]
        first_probs11.iloc[:, i+16] = model5_X_train_prob[:, i]
        first_probs11.iloc[:, i+20] = model6_X_train_prob[:, i]
        first_probs11.iloc[:, i+24] = model7_X_train_prob[:, i]
        first_probs11.iloc[:, i+28] = model8_X_train_prob[:, i]
        first_probs11.iloc[:, i+32] = model9_X_train_prob[:, i]
        first_probs11.iloc[:, i+36] = model10_X_train_prob[:, i]
        first_probs11.iloc[:, i+40] = model11_X_train_prob[:, i]

    test_probs11 = pd.DataFrame(np.zeros((1000, 11*1*4)))
    for i in range(4):
        test_probs11.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs11.iloc[:, i+4] = model2_X_test_prob[:, i]
        test_probs11.iloc[:, i+8] = model3_X_test_prob[:, i]
        test_probs11.iloc[:, i+12] = model4_X_test_prob[:, i]
        test_probs11.iloc[:, i+16] = model5_X_test_prob[:, i]
        test_probs11.iloc[:, i+20] = model6_X_test_prob[:, i]
        test_probs11.iloc[:, i+24] = model7_X_test_prob[:, i]
        test_probs11.iloc[:, i+28] = model8_X_test_prob[:, i]
        test_probs11.iloc[:, i+32] = model9_X_test_prob[:, i]
        test_probs11.iloc[:, i+36] = model10_X_test_prob[:, i]
        test_probs11.iloc[:, i+40] = model11_X_test_prob[:, i]

    del model11_X_train_prob
    del model11_X_test_prob

    # 書き込み
    np.save('first_probs11.npy', first_probs11)
    np.save('test_probs11.npy', test_probs11)

    first_probs10 = pd.DataFrame(np.zeros((25440, 10*1*4)))
    for i in range(4):
        first_probs10.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs10.iloc[:, i+4] = model2_X_train_prob[:, i]
        first_probs10.iloc[:, i+8] = model3_X_train_prob[:, i]
        first_probs10.iloc[:, i+12] = model4_X_train_prob[:, i]
        first_probs10.iloc[:, i+16] = model5_X_train_prob[:, i]
        first_probs10.iloc[:, i+20] = model6_X_train_prob[:, i]
        first_probs10.iloc[:, i+24] = model7_X_train_prob[:, i]
        first_probs10.iloc[:, i+28] = model8_X_train_prob[:, i]
        first_probs10.iloc[:, i+32] = model9_X_train_prob[:, i]
        first_probs10.iloc[:, i+36] = model10_X_train_prob[:, i]

    test_probs10 = pd.DataFrame(np.zeros((1000, 10*1*4)))
    for i in range(4):
        test_probs10.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs10.iloc[:, i+4] = model2_X_test_prob[:, i]
        test_probs10.iloc[:, i+8] = model3_X_test_prob[:, i]
        test_probs10.iloc[:, i+12] = model4_X_test_prob[:, i]
        test_probs10.iloc[:, i+16] = model5_X_test_prob[:, i]
        test_probs10.iloc[:, i+20] = model6_X_test_prob[:, i]
        test_probs10.iloc[:, i+24] = model7_X_test_prob[:, i]
        test_probs10.iloc[:, i+28] = model8_X_test_prob[:, i]
        test_probs10.iloc[:, i+32] = model9_X_test_prob[:, i]
        test_probs10.iloc[:, i+36] = model10_X_test_prob[:, i]

    del model10_X_train_prob
    del model10_X_test_prob

    # 書き込み
    np.save('first_probs10.npy', first_probs10)
    np.save('test_probs10.npy', test_probs10)

    first_probs9 = pd.DataFrame(np.zeros((25440, 9*1*4)))
    for i in range(4):
        first_probs9.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs9.iloc[:, i+4] = model2_X_train_prob[:, i]
        first_probs9.iloc[:, i+8] = model3_X_train_prob[:, i]
        first_probs9.iloc[:, i+12] = model4_X_train_prob[:, i]
        first_probs9.iloc[:, i+16] = model5_X_train_prob[:, i]
        first_probs9.iloc[:, i+20] = model6_X_train_prob[:, i]
        first_probs9.iloc[:, i+24] = model7_X_train_prob[:, i]
        first_probs9.iloc[:, i+28] = model8_X_train_prob[:, i]
        first_probs9.iloc[:, i+32] = model9_X_train_prob[:, i]

    test_probs9 = pd.DataFrame(np.zeros((1000, 9*1*4)))
    for i in range(4):
        test_probs9.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs9.iloc[:, i+4] = model2_X_test_prob[:, i]
        test_probs9.iloc[:, i+8] = model3_X_test_prob[:, i]
        test_probs9.iloc[:, i+12] = model4_X_test_prob[:, i]
        test_probs9.iloc[:, i+16] = model5_X_test_prob[:, i]
        test_probs9.iloc[:, i+20] = model6_X_test_prob[:, i]
        test_probs9.iloc[:, i+24] = model7_X_test_prob[:, i]
        test_probs9.iloc[:, i+28] = model8_X_test_prob[:, i]
        test_probs9.iloc[:, i+32] = model9_X_test_prob[:, i]

    del model9_X_train_prob
    del model9_X_test_prob

    # 書き込み
    np.save('first_probs9.npy', first_probs9)
    np.save('test_probs9.npy', test_probs9)

    first_probs8 = pd.DataFrame(np.zeros((25440, 8*1*4)))
    for i in range(4):
        first_probs8.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs8.iloc[:, i+4] = model2_X_train_prob[:, i]
        first_probs8.iloc[:, i+8] = model3_X_train_prob[:, i]
        first_probs8.iloc[:, i+12] = model4_X_train_prob[:, i]
        first_probs8.iloc[:, i+16] = model5_X_train_prob[:, i]
        first_probs8.iloc[:, i+20] = model6_X_train_prob[:, i]
        first_probs8.iloc[:, i+24] = model7_X_train_prob[:, i]
        first_probs8.iloc[:, i+28] = model8_X_train_prob[:, i]

    test_probs8 = pd.DataFrame(np.zeros((1000, 8*1*4)))
    for i in range(4):
        test_probs8.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs8.iloc[:, i+4] = model2_X_test_prob[:, i]
        test_probs8.iloc[:, i+8] = model3_X_test_prob[:, i]
        test_probs8.iloc[:, i+12] = model4_X_test_prob[:, i]
        test_probs8.iloc[:, i+16] = model5_X_test_prob[:, i]
        test_probs8.iloc[:, i+20] = model6_X_test_prob[:, i]
        test_probs8.iloc[:, i+24] = model7_X_test_prob[:, i]
        test_probs8.iloc[:, i+28] = model8_X_test_prob[:, i]

    del model8_X_train_prob
    del model8_X_test_prob

    # 書き込み
    np.save('first_probs8.npy', first_probs8)
    np.save('test_probs8.npy', test_probs8)

    first_probs7 = pd.DataFrame(np.zeros((25440, 7*1*4)))
    for i in range(4):
        first_probs7.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs7.iloc[:, i+4] = model2_X_train_prob[:, i]
        first_probs7.iloc[:, i+8] = model3_X_train_prob[:, i]
        first_probs7.iloc[:, i+12] = model4_X_train_prob[:, i]
        first_probs7.iloc[:, i+16] = model5_X_train_prob[:, i]
        first_probs7.iloc[:, i+20] = model6_X_train_prob[:, i]
        first_probs7.iloc[:, i+24] = model7_X_train_prob[:, i]

    test_probs7 = pd.DataFrame(np.zeros((1000, 7*1*4)))
    for i in range(4):
        test_probs7.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs7.iloc[:, i+4] = model2_X_test_prob[:, i]
        test_probs7.iloc[:, i+8] = model3_X_test_prob[:, i]
        test_probs7.iloc[:, i+12] = model4_X_test_prob[:, i]
        test_probs7.iloc[:, i+16] = model5_X_test_prob[:, i]
        test_probs7.iloc[:, i+20] = model6_X_test_prob[:, i]
        test_probs7.iloc[:, i+24] = model7_X_test_prob[:, i]

    del model7_X_train_prob
    del model7_X_test_prob

    # 書き込み
    np.save('first_probs7.npy', first_probs7)
    np.save('test_probs7.npy', test_probs7)

    first_probs6 = pd.DataFrame(np.zeros((25440, 6*1*4)))
    for i in range(4):
        first_probs6.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs6.iloc[:, i+4] = model2_X_train_prob[:, i]
        first_probs6.iloc[:, i+8] = model3_X_train_prob[:, i]
        first_probs6.iloc[:, i+12] = model4_X_train_prob[:, i]
        first_probs6.iloc[:, i+16] = model5_X_train_prob[:, i]
        first_probs6.iloc[:, i+20] = model6_X_train_prob[:, i]

    test_probs6 = pd.DataFrame(np.zeros((1000, 6*1*4)))
    for i in range(4):
        test_probs6.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs6.iloc[:, i+4] = model2_X_test_prob[:, i]
        test_probs6.iloc[:, i+8] = model3_X_test_prob[:, i]
        test_probs6.iloc[:, i+12] = model4_X_test_prob[:, i]
        test_probs6.iloc[:, i+16] = model5_X_test_prob[:, i]
        test_probs6.iloc[:, i+20] = model6_X_test_prob[:, i]

    del model6_X_train_prob
    del model6_X_test_prob

    # 書き込み
    np.save('first_probs6.npy', first_probs6)
    np.save('test_probs6.npy', test_probs6)

    first_probs5 = pd.DataFrame(np.zeros((25440, 5*1*4)))
    for i in range(4):
        first_probs5.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs5.iloc[:, i+4] = model2_X_train_prob[:, i]
        first_probs5.iloc[:, i+8] = model3_X_train_prob[:, i]
        first_probs5.iloc[:, i+12] = model4_X_train_prob[:, i]
        first_probs5.iloc[:, i+16] = model5_X_train_prob[:, i]

    test_probs5 = pd.DataFrame(np.zeros((1000, 5*1*4)))
    for i in range(4):
        test_probs5.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs5.iloc[:, i+4] = model2_X_test_prob[:, i]
        test_probs5.iloc[:, i+8] = model3_X_test_prob[:, i]
        test_probs5.iloc[:, i+12] = model4_X_test_prob[:, i]
        test_probs5.iloc[:, i+16] = model5_X_test_prob[:, i]

    del model5_X_train_prob
    del model5_X_test_prob

    # 書き込み
    np.save('first_probs5.npy', first_probs5)
    np.save('test_probs5.npy', test_probs5)

    first_probs4 = pd.DataFrame(np.zeros((25440, 4*1*4)))
    for i in range(4):
        first_probs4.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs4.iloc[:, i+4] = model2_X_train_prob[:, i]
        first_probs4.iloc[:, i+8] = model3_X_train_prob[:, i]
        first_probs4.iloc[:, i+12] = model4_X_train_prob[:, i]

    test_probs4 = pd.DataFrame(np.zeros((1000, 4*1*4)))
    for i in range(4):
        test_probs4.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs4.iloc[:, i+4] = model2_X_test_prob[:, i]
        test_probs4.iloc[:, i+8] = model3_X_test_prob[:, i]
        test_probs4.iloc[:, i+12] = model4_X_test_prob[:, i]

    del model4_X_train_prob
    del model4_X_test_prob

    # 書き込み
    np.save('first_probs4.npy', first_probs4)
    np.save('test_probs4.npy', test_probs4)

    first_probs3 = pd.DataFrame(np.zeros((25440, 3*1*4)))
    for i in range(4):
        first_probs3.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs3.iloc[:, i+4] = model2_X_train_prob[:, i]
        first_probs3.iloc[:, i+8] = model3_X_train_prob[:, i]

    test_probs3 = pd.DataFrame(np.zeros((1000, 3*1*4)))
    for i in range(4):
        test_probs3.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs3.iloc[:, i+4] = model2_X_test_prob[:, i]
        test_probs3.iloc[:, i+8] = model3_X_test_prob[:, i]

    del model3_X_train_prob
    del model3_X_test_prob

    # 書き込み
    np.save('first_probs3.npy', first_probs3)
    np.save('test_probs3.npy', test_probs3)

    first_probs2 = pd.DataFrame(np.zeros((25440, 2*1*4)))
    for i in range(4):
        first_probs2.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs2.iloc[:, i+4] = model2_X_train_prob[:, i]

    test_probs2 = pd.DataFrame(np.zeros((1000, 2*1*4)))
    for i in range(4):
        test_probs2.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs2.iloc[:, i+4] = model2_X_test_prob[:, i]

    del model2_X_train_prob
    del model2_X_test_prob

    # 書き込み
    np.save('first_probs2.npy', first_probs2)
    np.save('test_probs2.npy', test_probs2)
