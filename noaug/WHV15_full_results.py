from pyexpat.errors import XML_ERROR_EXTERNAL_ENTITY_HANDLING
import sys
print(sys.version)
print(sys.path)
import struct
import numpy as np
import os, cv2, zipfile, io, re, glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from PIL import Image
from tensorflow import keras
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
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import gc
import time
ut = time.time()

from efficientnet.keras import EfficientNetB0 as Net
from efficientnet.keras import EfficientNetB5 as Net
from efficientnet.keras import center_crop_and_resize, preprocess_input

from math import sqrt
from scipy.special import ndtri

# Now build your graph and train it
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import random

start = time.time()


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

num_classes = len(classes)

del img_dirs

# 画像サイズ
image_size = 256
image_size2 = 256

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
            image = Image.open(imgfile)
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

def im2array2(path):
    X2 = []
    y2 = []
    class_num = 0

    for class_name in classes:
        if class_num == num_classes : break
        imgfiles = glob.glob(path + class_name + "/*.bmp")
        for imgfile in imgfiles:
            # 画像読み込み
            image = Image.open(imgfile)
            # RGB変換
            image = image.convert('RGB')
            # リサイズ
            image = image.resize((image_size2, image_size2))
            # 画像から配列に変換
            data = np.asarray(image)
            X2.append(data)
            y2.append(classes.index(class_name))
        class_num += 1

    X2 = np.array(X2)
    y2 = np.array(y2)

    return X2, y2

##testデータ取得
X_test, y_test = im2array("hikakudata/Test/")
X_test2, y_test2 = im2array2("hikakudata/Test/")

##trainデータ取得
X_train, y_train = im2array("hikakudata/Train/")
X_train2, y_train2 = im2array2("hikakudata/Train/")

## データ型の変換
X_test = X_test.astype('float32')
X_test2 = X_test2.astype('float32')
X_train = X_train.astype('float32')
X_train2 = X_train2.astype('float32')


# 正規化
X_test /= 255
X_test2 /= 255
X_train /= 255
X_train2 /= 255


## one-hot 変換
y_test = to_categorical(y_test, num_classes = num_classes)
y_train = to_categorical(y_test, num_classes = num_classes)

##Top3の学習済みモデルと重みをロード
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
#model1_X_train_prob = model1.predict(X_train, verbose=1)
#model1_X_test_prob = model1.predict(X_test, verbose=1)

del f1

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
#model2_X_train_prob = model2.predict(X_train, verbose=1)
#model2_X_test_prob = model2.predict(X_test, verbose=1)
del f2

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
#model3_X_train_prob = model3.predict(X_train, verbose=1)
#model3_X_test_prob = model3.predict(X_test, verbose=1)
del f3

#del y_pred3
gc.collect()
print("Top3: 取り込み完了")

#Top4_SeResNeXt50_all
f4 = open(model_dir +"liver_256_noaugSeResNeXt50_all_20210729_rsf.json", 'r')
loaded_model_json4 = f4.read()
f4.close()
model4 = model_from_json(loaded_model_json4)
model4.load_weights(weights_dir + 'liver_256_noaugSeResNeXt50_all_20210729_rsf.h5')
#model4_X_train_prob = model4.predict(X_train, verbose=1)
#model4_X_test_prob = model4.predict(X_test, verbose=1)
del f4
#del model4
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
#model5_X_train_prob = model5.predict(X_train, verbose=1)
#model5_X_test_prob = model5.predict(X_test, verbose=1)
del f5
#del model5
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
#model6_X_train_prob = model4.predict(X_train, verbose=1)
#model6_X_test_prob = model4.predict(X_test, verbose=1)
del f6
#del model6
#del y_pred6
gc.collect()
print("Top6 : 取り込み完了")

#Top7_SeResNeXt101_all
f7 = open(model_dir +"liver_256_noaugSeResNeXt101_all_20220419_rsf_batch32.json", 'r')
loaded_model_json7 = f7.read()
f7.close()
model7 = model_from_json(loaded_model_json7)
model7.load_weights(weights_dir + 'liver_256_noaugSeResNeXt101_all_20220419_rsf_batch32.h5')
#model7_X_train_prob = model7.predict(X_train, verbose=1)
#model7_X_test_prob = model7.predict(X_test, verbose=1)
del f7
#del model7
#del y_pred7
gc.collect()
print("Top7 : 取り込み完了")

#Top8_ResNet101_26
f8 = open(model_dir +"liver_size256_noaugResNet101_26_rsf_batch32_20220423.json", 'r')
loaded_model_json8 = f8.read()
f8.close()
model8 = model_from_json(loaded_model_json8)
model8.load_weights(weights_dir + 'liver_size256_noaugResNet101_26_rsf_batch32_20220423.h5')
#model8_X_train_prob = model8.predict(X_train, verbose=1)
#model8_X_test_prob = model8.predict(X_test, verbose=1)
del f8
#del model8
#del y_pred8
gc.collect()
print("Top8 : 取り込み完了")

#Top9_ResNet50_all
f9 = open(model_dir +"liver_noaugResNet50_20210727_rsf_batch32_size256.json", 'r')
loaded_model_json9 = f9.read()
f9.close()
model9 = model_from_json(loaded_model_json9)
model9.load_weights(weights_dir + 'liver_noaugResNet50_20210727_rsf_batch32_size256.h5')
#model9_X_train_prob = model9.predict(X_train, verbose=1)
#model9_X_test_prob = model9.predict(X_test, verbose=1)
del f9
#del model9
gc.collect()
print("Top9 : 取り込み完了")


#Top10_InceptionV3_249
f10 = open(model_dir +"Amed_Liver_249_20210726InceptionV3_rsf_batch32_size256.json", 'r')
loaded_model_json10 = f10.read()
f10.close()
model10 = model_from_json(loaded_model_json10)
model10.load_weights(weights_dir + 'Amed_Liver_249_20210726InceptionV3_rsf_batch32_size256.h5')
#model10_X_train_prob = model10.predict(X_train, verbose=1)
#model10_X_test_prob = model10.predict(X_test, verbose=1)
del f10
#del model10
gc.collect()
print("Top10 : 取り込み完了")

#Top11_EfficientnetB2_NoisyStudent
f11 = open(model_dir +"Amed_Liver_-20_256_20220312efficientnetB2_rsf_batch32_noisy.json", 'r')
loaded_model_json11 = f11.read()
f11.close()
model11 = model_from_json(loaded_model_json11)
model11.load_weights(weights_dir + 'Amed_Liver_-20_256_20220312efficientnetB2_rsf_batch32_noisy.h5')
#model11_X_train_prob = model11.predict(X_train, verbose=1)
#model11_X_test_prob = model11.predict(X_test, verbose=1)
del f11
#del model11
gc.collect()
print("Top11 : 取り込み完了")

#Top12_EfficientnetB5_NoisyStudent
f12 = open(model_dir +"Amed_Liver_-20_256_20220423efficientnetB5_rsf_batch32_noisy.json", 'r')
loaded_model_json12 = f12.read()
f12.close()
model12 = model_from_json(loaded_model_json12)
model12.load_weights(weights_dir + 'Amed_Liver_-20_256_20220423efficientnetB5_rsf_batch32_noisy.h5')
#model12_X_train_prob = model12.predict(X_train, verbose=1)
#model12_X_test_prob = model12.predict(X_test, verbose=1)
del f12
#del model12
gc.collect()
print("Top12 : 取り込み完了")

#Top13_EfficientnetB3_NoisyStudent
f13 = open(model_dir +"Amed_Liver_-20_256_20220420efficientnetB3_rsf_batch32_noisy.json", 'r')
loaded_model_json13 = f13.read()
f13.close()
model13 = model_from_json(loaded_model_json13)
model13.load_weights(weights_dir + 'Amed_Liver_-20_256_20220420efficientnetB3_rsf_batch32_noisy.h5')
#model13_X_train_prob = model13.predict(X_train, verbose=1)
#model13_X_test_prob = model13.predict(X_test, verbose=1)
del f13
#del model13
gc.collect()
print("Top13 : 取り込み完了")

#Top14_EfficinetnetB6_NoisyStuden
f14 = open(model_dir +"Amed_Liver_-20_256_20220312efficientnetB6_rsf_batch32_noisy.json", 'r')
loaded_model_json14 = f14.read()
f14.close()
model14 = model_from_json(loaded_model_json14)
model14.load_weights(weights_dir + 'Amed_Liver_-20_256_20220312efficientnetB6_rsf_batch32_noisy.h5')
#model14_X_train_prob = model14.predict(X_train, verbose=1)
#model14_X_test_prob = model14.predict(X_test, verbose=1)
del f14
#del model14
gc.collect()
print("Top14 : 取り込み完了")


#Top15_EfficinetnetB1_NoisyStuden
f15 = open(model_dir +"Amed_Liver_-20_256_20220312efficientnetB1_rsf_noisy.json", 'r')
loaded_model_json15 = f15.read()
f15.close()
model15 = model_from_json(loaded_model_json15)
model15.load_weights(weights_dir + 'Amed_Liver_-20_256_20220312efficientnetB1_rsf_noisy.h5')
#model15_X_train_prob = model15.predict(X_train, verbose=1)
#model15_X_test_prob = model15.predict(X_test, verbose=1)
del f15
#del model15
gc.collect()
print("Top15 : 取り込み完了")

'''
#Top16_EfficinetnetB4_NoisyStuden
f16 = open(model_dir +"Amed_Liver_-20_256_20220312efficientnetB4_rsf_batch32_noisy.json", 'r')
loaded_model_json16 = f16.read()
f16.close()
model16 = model_from_json(loaded_model_json16)
model16.load_weights(weights_dir + 'Amed_Liver_-20_256_20220312efficientnetB4_rsf_batch32_noisy.h5')
#model16_X_train_prob = model16.predict(X_train, verbose=1)
#model16_X_test_prob = model16.predict(X_test, verbose=1)
del f16
#del model16
gc.collect()
print("Top16 : 取り込み完了")
'''

end_reading = time.time()

##################################
#それぞれの分類精度
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def Voting(x_voting,y_voting,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15):
    model1_predict = np.argmax(model1.predict(x_voting), axis = 1) 
    model2_predict = np.argmax(model2.predict(x_voting), axis = 1) 
    model3_predict = np.argmax(model3.predict(x_voting), axis = 1) 
    model4_predict = np.argmax(model4.predict(x_voting), axis = 1) 
    model5_predict = np.argmax(model5.predict(x_voting), axis = 1) 
    model6_predict = np.argmax(model6.predict(x_voting), axis = 1) 
    model7_predict = np.argmax(model7.predict(x_voting), axis = 1) 
    model8_predict = np.argmax(model8.predict(x_voting), axis = 1) 
    model9_predict = np.argmax(model9.predict(x_voting), axis = 1) 
    model10_predict = np.argmax(model10.predict(x_voting), axis = 1) 
    model11_predict = np.argmax(model11.predict(x_voting), axis = 1) 
    model12_predict = np.argmax(model12.predict(x_voting), axis = 1) 
    model13_predict = np.argmax(model13.predict(x_voting), axis = 1) 
    model14_predict = np.argmax(model14.predict(x_voting), axis = 1) 
    model15_predict = np.argmax(model15.predict(x_voting), axis = 1) 


    model1_onehot = to_categorical(model1_predict, 10)
    model2_onehot = to_categorical(model2_predict, 10)
    model3_onehot = to_categorical(model3_predict, 10)
    model4_onehot = to_categorical(model4_predict, 10)
    model5_onehot = to_categorical(model5_predict, 10)
    model6_onehot = to_categorical(model6_predict, 10)
    model7_onehot = to_categorical(model7_predict, 10)
    model8_onehot = to_categorical(model8_predict, 10)
    model9_onehot = to_categorical(model9_predict, 10)
    model10_onehot = to_categorical(model10_predict, 10)
    model11_onehot = to_categorical(model11_predict, 10)
    model12_onehot = to_categorical(model12_predict, 10)
    model13_onehot = to_categorical(model13_predict, 10)
    model14_onehot = to_categorical(model14_predict, 10)
    model15_onehot = to_categorical(model15_predict, 10)

    voting_result = []
    for i in range(len(x_voting)):
        voting = np.argmax(k1*model1_onehot[i] +
                            k2*model2_onehot[i] + k3*model3_onehot[i] +
                            k4*model4_onehot[i] + k5*model5_onehot[i] +
                            k6*model6_onehot[i] + k7*model7_onehot[i] +
                            k8*model8_onehot[i] + k9*model9_onehot[i] +
                            k10*model10_onehot[i] + k11*model11_onehot[i] +
                            k12*model12_onehot[i] + k13*model13_onehot[i] +
                            k14*model14_onehot[i] + k15*model15_onehot[i] )

        voting_result.append(voting)
        #print('voting_result:', voting_result)
    keras.backend.clear_session() # ←これです
    gc.collect()

    return voting_result


acc = accuracy_score(y_test2, Voting(X_test,y_test2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,))
print('Normal HardVoting16 acc:', acc)
weighted_acc =  accuracy_score(y_test2, Voting(X_test,y_test2,1.38609321, 1.22620827, 0.70743252, 1.44442645, 0.76942426, 1.45076163,
 0.83424487, 0.50770404, 0.79740872, 1.37535008, 0.74502187, 0.8149234,
 1.11670352, 0.50303698, 1.43144262))
print('WeightedHardVoting15 acc:', weighted_acc)
print(classification_report(y_test2, Voting(X_test,y_test2,1.38609321, 1.22620827, 0.70743252, 1.44442645, 0.76942426, 1.45076163,
 0.83424487, 0.50770404, 0.79740872, 1.37535008, 0.74502187, 0.8149234,
 1.11670352, 0.50303698, 1.43144262), digits=3, target_names=['BLTumor', 'LCyst', 'MLCancer', 'PLCancer']))
cf_matrix = confusion_matrix(y_test2, Voting(X_test,y_test2,1.38609321, 1.22620827, 0.70743252, 1.44442645, 0.76942426, 1.45076163,
 0.83424487, 0.50770404, 0.79740872, 1.37535008, 0.74502187, 0.8149234,
 1.11670352, 0.50303698, 1.43144262))
cf_matrix1 = pd.DataFrame(data=cf_matrix, index=['BLTumor', 'LCyst', 'MLCancer', 'PLCancer'], 
                           columns=['BLTumor', 'LCyst', 'MLCancer', 'PLCancer'])
print(cf_matrix1)

# testデータ n件の正解ラベル
#true_classes = np.argmax(y_test[0:n], axis = 1)
true_classes = np.argmax(y_test[0:1000], axis = 1)
#print('correct:', true_classes)
np.savetxt('correctLiver_WeightedHardVoting15.csv',true_classes,delimiter=',')

# testデータ n件の予測ラベル
#pred_classes = np.argmax(model.predict(X_test[0:n]), axis = 1)
pred_classes = Voting(X_test,y_test2,1.38609321, 1.22620827, 0.70743252, 1.44442645, 0.76942426, 1.45076163,
 0.83424487, 0.50770404, 0.79740872, 1.37535008, 0.74502187, 0.8149234,
 1.11670352, 0.50303698, 1.43144262)
#print('prediction:', pred_classes)
np.savetxt('predictionsLiver_WeightedHardVoting15.csv',pred_classes,delimiter=',')


sns.heatmap(cf_matrix, square=True, cbar=True, annot=True, cmap='Blues', fmt="d")
plt.yticks(rotation=0)
plt.xlabel("Predicted label", fontsize=13, rotation=0)
plt.ylabel("True label", fontsize=13)
#ax.set_ylim(len(cf_matrix), 0)
plt.savefig('WeightedHardVoting15_Heatmap_20220423_final_rsf.png')


# 0 = BLTumor, 1 = BLTumor, 2 = MLCancer, 3 = PLCancer
#print(classification_report(np.argmax(y_test, 1), np.argmax(y_pred_prob_lgb, 1), digits=3, target_names=['BLTumor', 'LCyst', 'MLCancer', 'PLCancer']))

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

f = open('WeightedHardVoting15_Full_Final_Result_amed_liver.txt', 'w')
f.write('WeightedHardVoting15_\n')
f.write(classification_report(y_test2, Voting(X_test,y_test2,1.38609321, 1.22620827, 0.70743252, 1.44442645, 0.76942426, 1.45076163,
 0.83424487, 0.50770404, 0.79740872, 1.37535008, 0.74502187, 0.8149234,
 1.11670352, 0.50303698, 1.43144262), digits=3, target_names=['BLTumor', 'LCyst', 'MLCancer', 'PLCancer']) + '\n')
f.write(' \n')
f.write('Macro Average\n')
f.write("Precision_MacroAve:" + str((conf_precision_BLT + conf_precision_LCyst + conf_precision_MLC + conf_precision_PLC)/4) + ", alpha = 95 CI for MacroAve:"+ str(tuple(precision_ci_MacroAve)) + '\n')
f.write("sensitivity_MacroAve:"+ str((conf_sensitivity_BLT + conf_sensitivity_LCyst + conf_sensitivity_MLC + conf_sensitivity_PLC)/4) + ",alpha = 95 CI for MacroAve:" + str(tuple(sensitivity_ci_MacroAve)) + '\n')
f.write("specificity_MacroAve:" + str((conf_specificity_BLT + conf_specificity_LCyst + conf_specificity_MLC + conf_specificity_PLC)/4) + ",alpha = 95 CI for MacroAve:" +  str(tuple(specificity_ci_MacroAve)) + '\n')
f.write("f1_score_MacroAve:" +str((conf_f1_score_BLT + conf_f1_score_LCyst + conf_f1_score_MLC + conf_f1_score_PLC)/4)  + '\n')
f.write(' \n')
f.write("Accuracy_MacroAve:" + str(conf_accuracy_total) + ",alpha = 95 CI for accuracy:" +  str(accuracy_ci_total) + ' \n')
f.write(' \n')
t1 = end_reading  -start
t2 = time.time()  -end_reading 
t3 = time.time() - start 

print('Read models time :', t1, 'sec')
print('WHV time = Total time -  reading models:', t2, 'sec')
print('Total time:', t3 ,'sec')
f.write('Read models time (sec):' + str(t1) + '\n')
f.write('WHV time = Total time -  reading models(sec):' + str(t2) + '\n')
f.write('Total time(sec) :' + str(t3) + '\n')
f.close()
