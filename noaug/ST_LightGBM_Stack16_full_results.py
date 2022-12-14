from __future__ import print_function, division
import sys
print(sys.version)
print(sys.path)
import struct
import numpy as np
import os, cv2, zipfile, io, re, glob
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
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

import shutil

import numpy as np 
from skimage.io import imread
from IPython.display import Image
from efficientnet.keras import EfficientNetB3 as Net
from efficientnet.keras import EfficientNetB0 as Net
from efficientnet.keras import center_crop_and_resize, preprocess_input
from tensorflow.keras.preprocessing import image

import lightgbm as lgb #LightGBM
from sklearn.model_selection import GridSearchCV

import gc

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

from sklearn.metrics import auc

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

# LightGBM Optuna??????????????????????????????????????????????????????
import optuna.integration.lightgbm as lgb

from math import sqrt
from scipy.special import ndtri


# Now build your graph and train it
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import random
#?????????seed????????????
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

#???????????????
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

    # ???????????????
    image_size = 256
    image_size2 = 300

    # ????????????????????????????????????
    def im2array(path):
        X = []
        y = []
        class_num = 0

        for class_name in classes:
            if class_num == num_classes : break
            imgfiles = glob.glob(path + class_name + "/*.bmp")
            for imgfile in imgfiles:
                # ??????????????????
                image = PIL.Image.open(imgfile)
                # RGB??????
                image = image.convert('RGB')
                # ????????????
                image = image.resize((image_size, image_size))
                # ???????????????????????????
                data = np.asarray(image)
                X.append(data)
                y.append(classes.index(class_name))
            class_num += 1

        X = np.array(X)
        y = np.array(y)

        return X, y
    ##test???????????????
    X_test, y_test = im2array("hikakudata/Test/")

    ## test?????????????????????
    X_test = X_test.astype('float32')

    # test?????????
    X_test /= 255

    ## test one-hot ??????
    y_test_old = y_test
    y_test = to_categorical(y_test, num_classes = num_classes)

    #train???????????????
    X_train, y_train = im2array("hikakudata/Train/")

    X_train = X_train.astype('float32')

    # train?????????
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

    # train one-hot ??????
    y_train_old = y_train
    y_train = to_categorical(y_train, num_classes = num_classes)

    # ??????????????????????????????????????????????????????


    #df_X_train  = pd.DataFrame(X_train)
    #df_X_test  = pd.DataFrame(X_test)
    
    #df_X_train  = X_train
    #df_X_test  = X_test
    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train_old  shape:", y_train_old.shape)
    print("pd_DataFrame_y_train shape:", pd.DataFrame(y_train))

    
    ##Top1_ResNeXt101?????????????????????????????????????????????
    models = []
    model_dir = './model/'
    weights_dir = './weights/'
    '''
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
    print("Top1: ??????????????????")
    

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
    print("Top2: ??????????????????")
    
    
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
    print("Top3: ??????????????????")
    
    
    ##Top4_
    f4 = open(model_dir +"Amed_Liver_-20_size256_20220312efficientnetB0_rsf_batch32_noisy.json", 'r')
    loaded_model_json4 = f4.read()
    f4.close()
    model4 = model_from_json(loaded_model_json4)
    model4.load_weights(weights_dir + 'Amed_Liver_-20_size256_20220312efficientnetB0_rsf_batch32_noisy.h5')
    #y_pred4 = model4.predict(X_train, verbose=1)
    #df_X4 = pd.DataFrame(y_pred4)
    model4_X_train_prob = model4.predict(X_train, verbose=1)
    model4_X_test_prob = model4.predict(X_test, verbose=1)
    del f4
    del model4
    #del y_pred4
    gc.collect()
    print("Top4 : ??????????????????")
    

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
    print("Top5 : ??????????????????")
    
    #Top6_SeResNeXt101_all
    f6 = open(model_dir +"liver_256_noaugSeResNeXt101_all_20220419_rsf_batch32.json", 'r')
    loaded_model_json6 = f6.read()
    f6.close()
    model6 = model_from_json(loaded_model_json6)
    model6.load_weights(weights_dir + 'liver_256_noaugSeResNeXt101_all_20220419_rsf_batch32.h5')
    model6_X_train_prob = model6.predict(X_train, verbose=1)
    model6_X_test_prob = model6.predict(X_test, verbose=1)
    del f6
    del model6
    #del y_pred5
    gc.collect()
    print("Top6 : ??????????????????")

    #Top7_ResNet101_26
    f7 = open(model_dir +"liver_size256_noaugResNet101_26_rsf_batch32_20220423.json", 'r')
    loaded_model_json7 = f7.read()
    f7.close()
    model7 = model_from_json(loaded_model_json7)
    model7.load_weights(weights_dir + 'liver_size256_noaugResNet101_26_rsf_batch32_20220423.h5')
    model7_X_train_prob = model7.predict(X_train, verbose=1)
    model7_X_test_prob = model7.predict(X_test, verbose=1)
    del f7
    del model7
    #del y_pred5
    gc.collect()
    print("Top7 : ??????????????????")

    #Top8_ResNet50_all
    f8 = open(model_dir +"liver_noaugResNet50_20210727_rsf_batch32_size256.json", 'r')
    loaded_model_json8 = f8.read()
    f8.close()
    model8 = model_from_json(loaded_model_json8)
    model8.load_weights(weights_dir + 'liver_noaugResNet50_20210727_rsf_batch32_size256.h5')
    model8_X_train_prob = model8.predict(X_train, verbose=1)
    model8_X_test_prob = model8.predict(X_test, verbose=1)
    del f8
    del model8
    gc.collect()
    print("Top8 : ??????????????????")

    #Top9_InceptionV3_249
    f9 = open(model_dir +"Amed_Liver_249_20210726InceptionV3_rsf_batch32_size256.json", 'r')
    loaded_model_json9 = f9.read()
    f9.close()
    model9 = model_from_json(loaded_model_json9)
    model9.load_weights(weights_dir + 'Amed_Liver_249_20210726InceptionV3_rsf_batch32_size256.h5')
    model9_X_train_prob = model9.predict(X_train, verbose=1)
    model9_X_test_prob = model9.predict(X_test, verbose=1)
    del f9
    del model9
    gc.collect()
    print("Top9 : ??????????????????")

    #Top10_EfficientnetB1_NoisyStudent
    f10 = open(model_dir +"Amed_Liver_-20_256_20220312efficientnetB1_rsf_noisy.json", 'r')
    loaded_model_json10 = f10.read()
    f10.close()
    model10 = model_from_json(loaded_model_json10)
    model10.load_weights(weights_dir + 'Amed_Liver_-20_256_20220312efficientnetB1_rsf_noisy.h5')
    model10_X_train_prob = model10.predict(X_train, verbose=1)
    model10_X_test_prob = model10.predict(X_test, verbose=1)
    del f10
    del model10
    gc.collect()
    print("Top10 : ??????????????????")


    #Top11_EfficientnetB5_NoisyStudent
    f11 = open(model_dir +"Amed_Liver_-20_256_20220423efficientnetB5_rsf_batch32_noisy.json", 'r')
    loaded_model_json11 = f11.read()
    f11.close()
    model11 = model_from_json(loaded_model_json11)
    model11.load_weights(weights_dir + 'Amed_Liver_-20_256_20220423efficientnetB5_rsf_batch32_noisy.h5')
    model11_X_train_prob = model11.predict(X_train, verbose=1)
    model11_X_test_prob = model11.predict(X_test, verbose=1)
    del f11
    del model11
    gc.collect()
    print("Top11 : ??????????????????")


    #Top12_EfficientnetB2_NoisyStudent
    f12 = open(model_dir +"Amed_Liver_-20_256_20220312efficientnetB2_rsf_batch32_noisy.json", 'r')
    loaded_model_json12 = f12.read()
    f12.close()
    model12 = model_from_json(loaded_model_json12)
    model12.load_weights(weights_dir + 'Amed_Liver_-20_256_20220312efficientnetB2_rsf_batch32_noisy.h5')
    model12_X_train_prob = model12.predict(X_train, verbose=1)
    model12_X_test_prob = model12.predict(X_test, verbose=1)
    del f12
    del model12
    gc.collect()
    print("Top12 : ??????????????????")

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
    print("Top13 : ??????????????????")

    #Top14_SeResNeXt50_all
    f14 = open(model_dir +"liver_256_noaugSeResNeXt50_all_20210729_rsf.json", 'r')
    loaded_model_json14 = f14.read()
    f14.close()
    model14 = model_from_json(loaded_model_json14)
    model14.load_weights(weights_dir + 'liver_256_noaugSeResNeXt50_all_20210729_rsf.h5')
    model14_X_train_prob = model14.predict(X_train, verbose=1)
    model14_X_test_prob = model14.predict(X_test, verbose=1)
    del f14
    del model14
    gc.collect()
    print("Top14 : ??????????????????")

    #Top15_EfficinetnetB6_NoisyStuden
    f15 = open(model_dir +"Amed_Liver_-20_256_20220312efficientnetB6_rsf_batch32_noisy.json", 'r')
    loaded_model_json15 = f15.read()
    f15.close()
    model15 = model_from_json(loaded_model_json15)
    model15_X_train_prob = model15.predict(X_train, verbose=1)
    model15_X_test_prob = model15.predict(X_test, verbose=1)
    del f15
    del model15
    gc.collect()
    print("Top15 : ??????????????????")

    #Top16_
    f16 = open(model_dir +"Amed_Liver_-20_256_20220312efficientnetB6_rsf_batch32_noisy.json", 'r')
    loaded_model_json16 = f16.read()
    f16.close()
    model16 = model_from_json(loaded_model_json16)
    model16.load_weights(weights_dir + 'Amed_Liver_-20_256_20220312efficientnetB6_rsf_batch32_noisy.h5')
    model16_X_train_prob = model16.predict(X_train, verbose=1)
    model16_X_test_prob = model16.predict(X_test, verbose=1)
    del f16
    del model16
    gc.collect()
    print("Top16 : ??????????????????")
    
    # ???????????????????????????????????????
    # ????????????????????????5?????????*5seed*4????????????
    first_probs = pd.DataFrame(np.zeros((25440, 16*1*4)))
    for i in range(4):
        first_probs.iloc[:, i] = model1_X_train_prob[:, i]
        first_probs.iloc[:, i+4] = model2_X_train_prob[:, i]
        first_probs.iloc[:, i+8] = model3_X_train_prob[:, i]
        first_probs.iloc[:, i+12] = model4_X_train_prob[:, i]
        first_probs.iloc[:, i+16] = model5_X_train_prob[:, i]
        first_probs.iloc[:, i+20] = model6_X_train_prob[:, i]
        first_probs.iloc[:, i+24] = model7_X_train_prob[:, i]
        first_probs.iloc[:, i+28] = model8_X_train_prob[:, i]
        first_probs.iloc[:, i+32] = model9_X_train_prob[:, i]
        first_probs.iloc[:, i+36] = model10_X_train_prob[:, i]
        first_probs.iloc[:, i+40] = model11_X_train_prob[:, i]
        first_probs.iloc[:, i+44] = model12_X_train_prob[:, i]
        first_probs.iloc[:, i+48] = model13_X_train_prob[:, i]
        first_probs.iloc[:, i+52] = model14_X_train_prob[:, i]
        first_probs.iloc[:, i+56] = model15_X_train_prob[:, i]
        first_probs.iloc[:, i+60] = model16_X_train_prob[:, i]

        
    test_probs = pd.DataFrame(np.zeros((1000, 16*1*4)))
    for i in range(4):
        test_probs.iloc[:, i] = model1_X_test_prob[:, i]
        test_probs.iloc[:, i+4] = model2_X_test_prob[:, i]
        test_probs.iloc[:, i+8] = model3_X_test_prob[:, i]
        test_probs.iloc[:, i+12] = model4_X_test_prob[:, i]
        test_probs.iloc[:, i+16] = model5_X_test_prob[:, i]
        test_probs.iloc[:, i+20] = model6_X_test_prob[:, i]
        test_probs.iloc[:, i+24] = model7_X_test_prob[:, i]
        test_probs.iloc[:, i+28] = model8_X_test_prob[:, i]
        test_probs.iloc[:, i+32] = model9_X_test_prob[:, i]
        test_probs.iloc[:, i+36] = model10_X_test_prob[:, i]
        test_probs.iloc[:, i+40] = model11_X_test_prob[:, i]
        test_probs.iloc[:, i+44] = model12_X_test_prob[:, i]
        test_probs.iloc[:, i+48] = model13_X_test_prob[:, i]
        test_probs.iloc[:, i+52] = model14_X_test_prob[:, i]
        test_probs.iloc[:, i+56] = model15_X_test_prob[:, i]
        test_probs.iloc[:, i+60] = model16_X_test_prob[:, i]


    del model1_X_train_prob
    del model2_X_train_prob
    del model3_X_train_prob
    del model4_X_train_prob
    del model5_X_train_prob
    del model6_X_train_prob
    del model7_X_train_prob
    del model8_X_train_prob
    del model9_X_train_prob
    del model10_X_train_prob
    del model11_X_train_prob
    del model12_X_train_prob
    del model13_X_train_prob
    del model14_X_train_prob
    del model15_X_train_prob
    del model16_X_train_prob

    
    del model1_X_test_prob
    del model2_X_test_prob
    del model3_X_test_prob
    del model4_X_test_prob
    del model5_X_test_prob
    del model6_X_test_prob
    del model7_X_test_prob
    del model8_X_test_prob
    del model9_X_test_prob
    del model10_X_test_prob
    del model11_X_test_prob
    del model12_X_test_prob
    del model13_X_test_prob
    del model14_X_test_prob
    del model15_X_test_prob
    del model16_X_test_prob
    
    # ????????????
    np.save('first_probs16.npy', first_probs)
    np.save('test_probs16.npy', test_probs)
    '''
    # ????????????
    first_probs = np.load ('first_probs16.npy')
    test_probs = np.load ('test_probs16.npy')


    print("????????????????????????Stack??????????????????")
    gc.collect()

    print(test_probs.shape)
        
        #first_probs.iloc[val_index, (seed_no * 2) + i] = model1_prob[:, i]
        #first_probs.iloc[val_index, (seed_no * 2) + 4 + i] = model2_prob[:, i]

        #first_probs.iloc[eval_cv_no, (seed_no * 3) + 8 + i] = model3_prob[:, i]

        #first_probs.iloc[eval_cv_no, (seed_no * 3) + 15 + i] = model4_prob[:, i]
        #first_probs.iloc[eval_cv_no, (seed_no * 3) + 20 + i] = model5_prob[:, i]         
    
    X_train2 = []
    X_test2 = []
    y_train2 = []
    y_test2 = []
    #df_y_train = pd.Series(y_train)

    print("????????????LightGBM????????????")
    
    from sklearn.model_selection import KFold
    FOLD = 5
    valid_scores = []

    kf = KFold(n_splits=FOLD, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(kf.split(first_probs, y_train_old)):
        X_train2, X_test2 = first_probs[train_index], first_probs[val_index]
        y_train2, y_test2 = y_train_old[train_index], y_train_old[val_index]

        # ???????????????????????????numpy???????????????
        test_preds = np.zeros((len(y_test), 5))

        # ???????????????????????????????????????0?????????????????????????????????
        row_no_list = list(range(len(y_train2)))

        # ???????????????????????????????????????
        lgb_train = lgb.Dataset(X_train2, y_train2)
        lgb_eval = lgb.Dataset(X_test2, y_test2) 
        lgb_test = lgb.Dataset(test_probs, y_test_old, reference=lgb_train)

        # LightGBM parameters
        params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'multiclass', # ?????? : ?????????????????? 
                'num_class': 4, # ???????????? : 4
                'metric': 'multi_logloss', # ???????????? : ?????????(= 1-?????????) 
                #?????????'multi_logloss'??????
                'learning_rate': 0.05,
                'verbose': 0,
                'deterministic':True, #????????????????????????????????????
                'force_row_wise':True  #????????????????????????????????????,
        }

        booster = lgb.LightGBMTuner(
            params = params, 
            train_set = lgb_train,
            valid_sets=lgb_eval,
            optuna_seed=123, #????????????????????????????????????
            num_boost_round=1000,
            early_stopping_rounds=50,
        )
        # ???????????????
        booster.run()

        # ????????????????????????????????????????????????
        booster.best_params

        # ????????????????????????????????????Booster????????????????????????????????????
        best_booster = booster.get_best_booster()
        print(f'fold {fold}')

    # ??????????????????????????? ((??????????????????????????? [?????????0???????????????,?????????1???????????????,?????????2???????????????] ?????????))
    #y_pred_prob_lgb = gbm_optuna.predict(test_probs, num_iteration=gbm_optuna.best_iteration)
    y_pred_prob_lgb = best_booster.predict(test_probs)
    # ??????????????????????????? (???????????????(0 or 1 or 2)?????????)
    #y_pred_lgb = np.argmax(y_pred_prob_lgb, axis=1) # ????????????????????????????????????????????????????????????
    y_pred_lgb = np.argmax(y_pred_prob_lgb, axis=1)


    # test????????? n?????????????????????
    #true_classes = np.argmax(y_test[0:n], axis = 1)
    true_classes = np.argmax(y_test[0:1000], axis = 1)
    print('correct:', true_classes)
    np.savetxt('correct_Amed_Liver_LightGBM_Stack 16models Trial1 Optuna_seed.csv',true_classes,delimiter=',')

    # test????????? n?????????????????????
    #pred_classes = np.argmax(model.predict(X_test[0:n]), axis = 1)
    pred_classes = np.argmax(y_pred_prob_lgb[0:1000], axis = 1)
    print('prediction:', pred_classes)
    np.savetxt('predictions_Amed_LightGBM_Stack 16models Trial1 Optuna_seed.csv',pred_classes,delimiter=',')

    print("4x4 EnsemgleStack_LightGBM_Stack 16models Trial1 Optuna seed_kfold_Report")
    print(booster.best_params)    
    cf_matrix = confusion_matrix(np.argmax(y_test, 1), y_pred_lgb)
    cf_matrix2 = pd.DataFrame(data=cf_matrix, index=['BLTumor', 'LCyst', 'MLCancer', 'PLCancer'], 
                            columns=['BLTumor', 'LCyst', 'MLCancer', 'PLCancer'])
    np.save('cf_matrix_EnsembleStack_LGBM16.npy', cf_matrix)
    print(cf_matrix2)
    print(classification_report(np.argmax(y_test, 1),y_pred_lgb, digits=3, target_names=['BLTumor', 'LCyst', 'MLCancer', 'PLCancer']))
    #acc = accuracy_score(y_test, y_pred_max)

    #HeatMap
    sns.heatmap(cf_matrix, square=True, cbar=True, annot=True, cmap='Blues', fmt="d")
    plt.yticks(rotation=0)
    plt.xlabel("Predicted label", fontsize=13, rotation=0)
    plt.ylabel("True label", fontsize=13)
    #ax.set_ylim(len(cf_matrix), 0)
    plt.savefig('EnsembleStack_LightGBM_Stack16_Optuna__Heatmap_20220423_seed.png')


    # 0 = BLTumor, 1 = BLTumor, 2 = MLCancer, 3 = PLCancer
    #print(classification_report(np.argmax(y_test, 1), np.argmax(y_pred_prob_lgb, 1), digits=3, target_names=['BLTumor', 'LCyst', 'MLCancer', 'PLCancer']))

    #ROC
    from sklearn.metrics import roc_curve, auc
    n_classes = 4
    fpr = {}
    tpr = {}
    roc_auc = {}
    p={}
    y_pred = y_pred_prob_lgb
    y_pred=np.array(y_pred)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        p[i] = roc_curve(y_test[:, i], y_pred[:, i])

    plt.plot(fpr[0], tpr[0], label=f'BLTumor', linestyle = "solid", c='black')
    plt.plot(fpr[1], tpr[1], label=f'LCyst', linestyle = "dashed", c='black')
    plt.plot(fpr[2], tpr[2], label=f'MLCancer', linestyle = "dashdot", c='black')
    plt.plot(fpr[3], tpr[3], label=f'PLCancer', linestyle = "dotted", c='black')

    plt.legend()
    plt.xlabel('1-specificity (FPR)') 
    plt.ylabel('sensitivity (TPR)')
    #plt.title('ROC Curve')
    #Replace USER with the user name of the computer you are using.

    plt.savefig('/home/USER/amed_liver/noaug/EnsembleStack_LightGBM_Stack16_Optuna__roc_curve1_seed.png')
    plt.gca().clear()

    #Micro&Macro??????

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr = mean_tpr / 4
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    plt.plot(fpr[0], tpr[0], label=f'BLTumor', linestyle = "solid", c='black')
    plt.plot(fpr[1], tpr[1], label=f'LCyst', linestyle = "dashed", c='black')
    plt.plot(fpr[2], tpr[2], label=f'MLCancer', linestyle = "dashdot", c='black')
    plt.plot(fpr[3], tpr[3], label=f'PLCancer', linestyle = "dotted", c='black')
    plt.plot(fpr['macro'], tpr['macro'], label='macro', lw=3, c='black')
    plt.legend()
    #Replace USER with the user name of the computer you are using.
    plt.savefig('/home/USER/amed_liver/noaug/EnsembleStack_LightGBM_Stack16_Optuna__roc_curve2_macro_seed.png')
    plt.gca().clear()

    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.plot(fpr[0], tpr[0], label=f'BLTumor', linestyle = "solid", c='black')
    plt.plot(fpr[1], tpr[1], label=f'LCyst', linestyle = "dashed", c='black')
    plt.plot(fpr[2], tpr[2], label=f'MLCancer', linestyle = "dashdot", c='black')
    plt.plot(fpr[3], tpr[3], label=f'PLCancer', linestyle = "dotted", c='black')
    plt.plot(fpr['macro'], tpr['macro'], label='macro', lw=3, c='gray')
    plt.plot(fpr['micro'], tpr['micro'], label='micro', lw=2, c='gray')
    plt.legend()
    #plt.show()
    plt.savefig('/home/USER/amed_liver/noaug/EnsembleStack_LightGBM_Stack16_Optuna_roc_curve3_both_seed.png')
    plt.gca().clear()

    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.plot(fpr[0], tpr[0], label=f'BLTumor', linestyle = "solid", c='black')
    plt.plot(fpr[1], tpr[1], label=f'LCyst', linestyle = "dashed", c='black')
    plt.plot(fpr[2], tpr[2], label=f'MLCancer', linestyle = "dashdot", c='black')
    plt.plot(fpr[3], tpr[3], label=f'PLCancer', linestyle = "dotted", c='black')
    plt.plot(fpr['micro'], tpr['micro'], label='micro', lw=3, c='gray')
    plt.legend()
    plt.savefig('/home/USER/amed_liver/noaug/EnsembleStack_LightGBM_Stack16_Optuna_roc_curve4_micro_seed.png')
    #plt.show()
    print('BLTumor_AUC:',auc(fpr[0], tpr[0]))
    print('LCyst_AUC:',auc(fpr[1], tpr[1]))
    print('MLCancer_AUC:',auc(fpr[2], tpr[2]))
    print('PLCancer_AUC:',auc(fpr[3], tpr[3]))
    print('Micro_AUC:',auc(fpr['micro'], tpr['micro']))
    print('Macro_AUC:',auc(fpr['macro'], tpr['macro']))

    #cf_matrix = cf_matrix.drop(cf_matrix.columns[[0]], axis=0)
    #cf_matrix = cf_matrix.drop(cf_matrix.columns[[0]], axis=0)


#CI 

from scipy.stats import norm

def ci(tp, n, alpha=0.05):
    """ Estimates confidence interval for Bernoulli p
    Args:
      tp: number of positive outcomes, TP in this case
      n: number of attemps, TP+FP for Precision, TP+FN for Recall
      alpha: confidence level
    Returns:
      Tuple[float, float]: lower and upper bounds of the confidence interval
    """
    p_hat = float(tp) / n
    z_score = norm.isf(alpha * 0.5)  # two sides, so alpha/2 on each side
    variance_of_sum = p_hat * (1-p_hat) / n
    std = variance_of_sum ** 0.5
    return p_hat - z_score * std, p_hat + z_score * std,  z_score * std


def acc_precision_recall_and_specificity_f1_score_ci(TP, FP, FN, TN, alpha=0.95):
    """Compute confidence intervals for sensitivity and specificity using Wilson's method. 
    
    This method does not rely on a normal approximation and results in accurate 
    confidence intervals even for small sample sizes.
    
    Parameters
    ----------
    TP : int
        Number of true positives
    FP : int 
        Number of false positives
    FN : int
        Number of false negatives
    TN : int
        Number of true negatives
    alpha : float, optional
        Desired confidence. Defaults to 0.95, which yields a 95% confidence interval. 
    
    Returns
    -------
    sensitivity_point_estimate : float
        Numerical estimate of the test sensitivity
    specificity_point_estimate : float
        Numerical estimate of the test specificity
    sensitivity_ci : Tuple (float, float)
        Lower and upper bounds on the alpha confidence interval for sensitivity
    specificity_ci
        Lower and upper bounds on the alpha confidence interval for specificity 
        
    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman, 
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000. 
    [2] E. B. Wilson, Probable inference, the law of succession, and statistical inference,
    J Am Stat Assoc 22:209-12, 1927. 
    """
    

    # Compute accuracy confidence intervals using method described in [1]
    #accuracy_ci = _proportion_ci(TP + TN, TP + FP + TN + FN, z)
    accuracy_ci = ci(TP + TN, TP + FP + TN + FN,)
    # Compute precision confidence intervals using method described in [1]
    precision_point_estimate = TP/(TP + FP)    
    #precision_ci = _proportion_ci(TP, TP + FP, z)
    precision_ci = ci(TP, TP + FP,)
    # Compute sensitivity(recall) confidence intervals using method described in [1]
    recall_point_estimate = TP/(TP + FN)    
    #sensitivity_ci = _proportion_ci(TP, TP + FN, z)
    sensitivity_ci = ci(TP, TP + FN,)
    # Compute specificity confidence intervals using method described in [1]
    #specificity_ci = _proportion_ci(TN, TN + FP, z)
    specificity_ci = ci(TN, TN + FP,)
    # Compute f1_score confidence intervals using method described in [1]
    #f1_score_ci = _proportion_ci(2*recall_point_estimate*precision_point_estimate, precision_point_estimate + precision_point_estimate, z)
    f1_score_ci = ci(2*recall_point_estimate*precision_point_estimate, recall_point_estimate + precision_point_estimate,)
    
    return accuracy_ci, precision_ci, sensitivity_ci, specificity_ci, f1_score_ci

matrix1 = cf_matrix[0][0]
matrix2 = cf_matrix[0][1]
matrix3 = cf_matrix[0][2]
matrix4 = cf_matrix[0][3]
matrix13 = cf_matrix[3][0]
matrix14 = cf_matrix[3][1]
matrix15 = cf_matrix[3][2]
matrix15 = cf_matrix[3][3]

print(matrix1)
print(matrix2)
print(matrix3)
print(matrix4)
print(matrix13)
print(matrix14)
print(matrix15)
print(matrix15)



#Benign Liver Tumor (BLT)
TP_BLT = cf_matrix[0][0]
TN_BLT = cf_matrix[1][1] + cf_matrix[1][2] + cf_matrix[1][3] + cf_matrix[2][1] + cf_matrix[2][2] + cf_matrix[2][3] +  cf_matrix[3][1] +  cf_matrix[3][2] +  cf_matrix[3][3] 
FP_BLT = cf_matrix[1][0] + cf_matrix[2][0] + cf_matrix[3][0] 
FN_BLT = cf_matrix[0][1] + cf_matrix[0][2] + cf_matrix[0][3]

# calculate accuracy
conf_accuracy_BLT = (float(TP_BLT + TN_BLT) / float(TP_BLT + FP_BLT + FN_BLT + TN_BLT))
# calculate precision
conf_precision_BLT = (TP_BLT / float(TP_BLT + FP_BLT))
# calculate the sensitivity(recall)
conf_sensitivity_BLT = (TP_BLT / float(TP_BLT + FN_BLT))
# calculate the specificity
conf_specificity_BLT = (TN_BLT / float(TN_BLT + FP_BLT))

# calculate f_1 score
conf_f1_score_BLT = float(2*conf_precision_BLT * conf_sensitivity_BLT) / float(conf_precision_BLT + conf_sensitivity_BLT)

accuracy_ci_BLT, precision_ci_BLT, sensitivity_ci_BLT, specificity_ci_BLT, f1_score_ci_BLT = acc_precision_recall_and_specificity_f1_score_ci(TP_BLT, FP_BLT, FN_BLT, TN_BLT, alpha=0.95)

print("Precision_BLT:",conf_precision_BLT, ",alpha = 95 CI for precision:", precision_ci_BLT)
print("Sensitivity_BLT:",conf_sensitivity_BLT, ",alpha = 95 CI for sensitivity:", sensitivity_ci_BLT)
print("Specificity_BLT:",conf_specificity_BLT, ",alpha = 95 CI for specificity", specificity_ci_BLT)
print("f1_score_BLT:",conf_f1_score_BLT )

#Liver Cyst (LCyst)
TP_LCyst = cf_matrix[1][1]
TN_LCyst = cf_matrix[0][0] + cf_matrix[0][2] + cf_matrix[0][3] + cf_matrix[2][0] + cf_matrix[3][0] + cf_matrix[2][2] +  cf_matrix[2][3] +  cf_matrix[3][2] +  cf_matrix[3][3] 
FP_LCyst = cf_matrix[0][1] + cf_matrix[2][1] + cf_matrix[3][1] 
FN_LCyst = cf_matrix[1][0] + cf_matrix[1][2] + cf_matrix[1][3]

# calculate accuracy
conf_accuracy_LCyst = (float(TP_LCyst + TN_LCyst) / float(TP_LCyst + FP_LCyst + FN_LCyst + TN_LCyst))
# calculate precision
conf_precision_LCyst = (TP_LCyst / float(TP_LCyst + FP_LCyst))
# calculate the sensitivity(recall)
conf_sensitivity_LCyst = (TP_LCyst / float(TP_LCyst + FN_LCyst))
# calculate the specificity
conf_specificity_LCyst = (TN_LCyst / float(TN_LCyst + FP_LCyst))
# calculate f_1 score
conf_f1_score_LCyst = float(2*conf_precision_LCyst * conf_sensitivity_LCyst) / float(conf_precision_LCyst + conf_sensitivity_LCyst)

accuracy_ci_LCyst, precision_ci_LCyst, sensitivity_ci_LCyst, specificity_ci_LCyst, f1_score_ci_LCyst = acc_precision_recall_and_specificity_f1_score_ci(TP_LCyst, FP_LCyst, FN_LCyst, TN_LCyst, alpha=0.95)

print("Precision_LCyst:",conf_precision_LCyst, ",alpha = 95 CI for precision:", precision_ci_LCyst)
print("Sensitivity_LCyst:",conf_sensitivity_LCyst, ",alpha = 95 CI for sensitivity:", sensitivity_ci_LCyst)
print("Specificity_LCyst:",conf_specificity_LCyst, ",alpha = 95 CI for specificity", specificity_ci_LCyst)
print("f1_score_LCyst:",conf_f1_score_LCyst )

#Metastatic Liver Cancer (MLC)
TP_MLC = cf_matrix[2][2]
TN_MLC = cf_matrix[3][3] + cf_matrix[0][0] + cf_matrix[0][1] + cf_matrix[1][0] + cf_matrix[1][1] + cf_matrix[0][3] +  cf_matrix[1][3] +  cf_matrix[3][0] +  cf_matrix[3][1] 
FP_MLC = cf_matrix[0][2] + cf_matrix[1][2] + cf_matrix[3][2] 
FN_MLC = cf_matrix[2][0] + cf_matrix[2][1] + cf_matrix[2][3]

# calculate accuracy
conf_accuracy_MLC = (float(TP_MLC + TN_MLC) / float(TP_MLC + FP_MLC + FN_MLC + TN_MLC))
# calculate precision
conf_precision_MLC = (TP_MLC / float(TP_MLC + FP_MLC))
# calculate the sensitivity(recall)
conf_sensitivity_MLC = (TP_MLC / float(TP_MLC + FN_MLC))
# calculate the specificity
conf_specificity_MLC = (TN_MLC / float(TN_MLC + FP_MLC))
# calculate f_1 score
conf_f1_score_MLC = float(2*conf_precision_MLC * conf_sensitivity_MLC) / float(conf_precision_MLC + conf_sensitivity_MLC)

accuracy_ci_MLC, precision_ci_MLC, sensitivity_ci_MLC, specificity_ci_MLC, f1_score_ci_MLC = acc_precision_recall_and_specificity_f1_score_ci(TP_MLC, FP_MLC, FN_MLC, TN_MLC, alpha=0.95)

print("Precision_MLC:",conf_precision_MLC, ",alpha = 95 CI for precision:", precision_ci_MLC)
print("Sensitivity_MLC:",conf_sensitivity_MLC, ",alpha = 95 CI for sensitivity:", sensitivity_ci_MLC)
print("Specificity_MLC:",conf_specificity_MLC, ",alpha = 95 CI for specificity", specificity_ci_MLC)
print("f1_score_MLC:",conf_f1_score_MLC)

#Primary Liver Cancer (PLC)
TP_PLC = cf_matrix[3][3]
TN_PLC = cf_matrix[0][0] + cf_matrix[0][1] + cf_matrix[0][2] + cf_matrix[1][0] + cf_matrix[1][1] + cf_matrix[1][2] +  cf_matrix[2][0] +  cf_matrix[2][1] +  cf_matrix[2][2] 
FP_PLC = cf_matrix[0][3] + cf_matrix[1][3] + cf_matrix[2][3] 
FN_PLC = cf_matrix[3][0] + cf_matrix[3][1] + cf_matrix[3][2]


#print("TP:", TP_PLC)
#print("TN:", TN_PLC)
#print("FP:", FP_PLC)
#print("FN:", FN_PLC)
#print("Total:", TP_PLC+TN_PLC+FP_PLC+FN_PLC)

# calculate accuracy
conf_accuracy_PLC = (float(TP_PLC + TN_PLC) / float(TP_PLC + FP_PLC + FN_PLC + TN_PLC))
# calculate precision
conf_precision_PLC = (TP_PLC / float(TP_PLC + FP_PLC))
#print ("PLC precision:", conf_precision_PLC, "(", TP_PLC, "/", float(TP_PLC + FP_PLC), ")")
# calculate the sensitivity(recall)
conf_sensitivity_PLC = (TP_PLC / float(TP_PLC + FN_PLC))
#print ("PLC recall(sensitivity):", conf_sensitivity_PLC, "(", TP_PLC, "/", float(TP_PLC + FN_PLC), ")")
# calculate the specificity
conf_specificity_PLC = (TN_PLC / float(TN_PLC + FP_PLC))
#print ("PLC specificity:", conf_specificity_PLC, "(", TN_PLC, "/", float(TN_PLC + FP_PLC), ")")
# calculate f_1 score
conf_f1_score_PLC = float(2*conf_precision_PLC * conf_sensitivity_PLC) / float(conf_precision_PLC + conf_sensitivity_PLC)
#print ("PLC f_1 score:", conf_f1_score_PLC, "(", 2 * (conf_precision_PLC * conf_sensitivity_PLC), "/", conf_precision_PLC + conf_sensitivity_PLC, ")")

accuracy_ci_PLC, precision_ci_PLC, sensitivity_ci_PLC, specificity_ci_PLC, f1_score_ci_PLC = acc_precision_recall_and_specificity_f1_score_ci(TP_PLC, FP_PLC, FN_PLC, TN_PLC, alpha=0.95)

print("Precision_PLC:",conf_precision_PLC, ",alpha = 95 CI for precision:", precision_ci_PLC)
print("Sensitivity_PLC:",conf_sensitivity_PLC, ",alpha = 95 CI for sensitivity:", sensitivity_ci_PLC)
print("Specificity_PLC:",conf_specificity_PLC, ",alpha = 95 CI for specificity", specificity_ci_PLC)
print("f1_score_PLC:",conf_f1_score_PLC )

#Macro Ave 
precision_ci_MacroAve = [(x1 + x2 + x3 + x4)/4 for (x1, x2, x3, x4) in zip(list(precision_ci_BLT), list(precision_ci_LCyst), list(precision_ci_MLC), list(precision_ci_PLC))]
sensitivity_ci_MacroAve = [(x1 + x2 + x3 + x4)/4 for (x1, x2, x3, x4) in zip(list(sensitivity_ci_BLT), list(sensitivity_ci_LCyst), list(sensitivity_ci_MLC), list(sensitivity_ci_PLC))]
specificity_ci_MacroAve = [(x1 + x2 + x3 + x4)/4 for (x1, x2, x3, x4) in zip(list(specificity_ci_BLT), list(specificity_ci_LCyst), list(specificity_ci_MLC), list(specificity_ci_PLC))]
f1_score_ci_MacroAve = [(x1 + x2 + x3 + x4)/4 for (x1, x2, x3, x4) in zip(list(f1_score_ci_BLT), list(f1_score_ci_LCyst), list(f1_score_ci_MLC), list(f1_score_ci_PLC))]
print("Macro Average")
print("Precision_MacroAve:",(conf_precision_BLT + conf_precision_LCyst + conf_precision_MLC + conf_precision_PLC) / 4 , ",alpha = 95 CI for MacroAve:", tuple(precision_ci_MacroAve))
print("sensitivity_MacroAve:",(conf_sensitivity_BLT + conf_sensitivity_LCyst + conf_sensitivity_MLC + conf_sensitivity_PLC) / 4 , ",alpha = 95 CI for MacroAve:", tuple(sensitivity_ci_MacroAve))
print("specificity_MacroAve:",(conf_specificity_BLT + conf_specificity_LCyst + conf_specificity_MLC + conf_specificity_PLC) / 4 , ",alpha = 95 CI for MacroAve:",  tuple(specificity_ci_MacroAve))
print("f1_score_MacroAve:",(conf_f1_score_BLT + conf_f1_score_LCyst + conf_f1_score_MLC + conf_f1_score_PLC) / 4)

#Accuracy + 95%CI
TPTN = cf_matrix[0][0] + cf_matrix[1][1] + cf_matrix[2][2] + cf_matrix[3][3]
alpha=0.95
z1 = -ndtri((1.0-alpha)/2)
accuracy_ci_total = ci(TPTN, 1000,)
conf_accuracy_total = TPTN/1000
print("Accuracy_MacroAve:", conf_accuracy_total, ",alpha = 95 CI for accuracy:", accuracy_ci_total)
