from __future__ import print_function
import keras
import sys
print(sys.version)
print(sys.path)
import struct
import numpy as np
import os, cv2, zipfile, io, re, glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import PIL.Image
#from PIL import Image
from keras import layers, models
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

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from IPython.display import Image
from efficientnet.keras import EfficientNetB0 as Net
from efficientnet.keras import center_crop_and_resize, preprocess_input
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf
from efficientnet.keras import EfficientNetB0
from math import sqrt
from scipy.special import ndtri

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

num_classes = len(classes)

del img_dirs

# 画像サイズ
image_size = 256

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

## データ型の変換
X_test = X_test.astype('float32')

# 正規化
X_test /= 255

## one-hot 変換
y_test = to_categorical(y_test, num_classes = num_classes)

## efficientnetB0 学習済みモデルと重みをロード
model_dir = './model/'
weights_dir = './weights/'
    ## Fully-connected層（FC）はいらないのでinclude_top=False）
#    input_tensor = Input(shape=(img_rows, img_cols, 3))
##   vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
f = open(model_dir +"Amed_Liver_-20_256_20220312efficientnetB0_rsf_noisy.json", 'r')
loaded_model_json = f.read()
f.close()
model = model_from_json(loaded_model_json)

#model = model_from_json(open('chest_finetunig2.json').read())
model.load_weights(weights_dir + 'Amed_Liver_-20_256_20220312efficientnetB0_rsf_noisy.h5')


#評価　一覧
y_pred = model.predict(X_test, verbose=1)
y_pred_keras = model.predict(X_test, verbose=1)


# testデータ n件の正解ラベル
#true_classes = np.argmax(y_test[0:n], axis = 1)
true_classes = np.argmax(y_test[0:1000], axis = 1)
print('correct:', true_classes)
np.savetxt('correctLiver_Liver_-20_256_20220312efficientnetB0_rsf_batch32_noisy.csv',true_classes,delimiter=',')

# testデータ n件の予測ラベル
#pred_classes = np.argmax(model.predict(X_test[0:n]), axis = 1)
pred_classes = np.argmax(model.predict(X_test[0:1000]), axis = 1)
print('prediction:', pred_classes)
np.savetxt('predictionsLiver_Liver_-20_256_20220312efficientnetB0_rsf_batch32_noisy.csv',pred_classes,delimiter=',')

#Parameters
model.summary()

print('4x4_Liver_-20_256_20220312efficientnetB0_rsf_batch32_noisy_Results')

cf_matrix = confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1))
cf_matrix1 = pd.DataFrame(data=cf_matrix, index=['BLTumor', 'LCyst', 'MLCancer', 'PLCancer'], 
                           columns=['BLTumor', 'LCyst', 'MLCancer', 'PLCancer'])
print(cf_matrix)

sns.heatmap(cf_matrix, square=True, cbar=True, annot=True, cmap='Blues', fmt="d")
plt.yticks(rotation=0)
plt.xlabel("Predicted label", fontsize=13, rotation=0)
plt.ylabel("True label", fontsize=13)
#ax.set_ylim(len(cf_matrix), 0)
plt.savefig('Liver_-20_256_20220312efficientnetB0_rsf_batch32_noisy_Heatmap.png')

print()
# 0 = BLTumor, 1 = BLTumor, 2 = MLCancer, 3 = PLCancer
print(classification_report(np.argmax(y_test, 1), np.argmax(y_pred, 1), digits=3, target_names=['BLTumor', 'LCyst', 'MLCancer', 'PLCancer']))

#ROC
from sklearn.metrics import roc_curve, auc
n_classes = 4
fpr = {}
tpr = {}
roc_auc = {}
p={}
#y_pred=np.array(y_pred_soft15)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    p[i] = roc_curve(y_test[:, i], y_pred[:, i])
import matplotlib.pyplot as plt


plt.plot(fpr[0], tpr[0], label=f'BLTumor', linestyle = "solid", c='black')
plt.plot(fpr[1], tpr[1], label=f'LCyst', linestyle = "dashed", c='black')
plt.plot(fpr[2], tpr[2], label=f'MLCancer', linestyle = "dashdot", c='black')
plt.plot(fpr[3], tpr[3], label=f'PLCancer', linestyle = "dotted", c='black')

plt.legend()
plt.xlabel('1-specificity (FPR)') 
plt.ylabel('sensitivity (TPR)')
#plt.title('ROC Curve')
#plt.show()
plt.savefig('/home/imedix/amed_liver/noaug/efficientnetB0_rsf_batch32_liver_noaug_roc_curve1.png')
plt.gca().clear()

#Micro&Macro平均
import numpy as np
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
#plt.show()
plt.savefig('/home/imedix/amed_liver/noaug/efficientnetB0_rsf_batch32_liver_noaug_roc_curve2_macro.png')
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
plt.savefig('/home/imedix/amed_liver/noaug/efficientnetB0_rsf_batch32_liver_noaug_roc_curve3_both.png')
plt.gca().clear()

fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.plot(fpr[0], tpr[0], label=f'BLTumor', linestyle = "solid", c='black')
plt.plot(fpr[1], tpr[1], label=f'LCyst', linestyle = "dashed", c='black')
plt.plot(fpr[2], tpr[2], label=f'MLCancer', linestyle = "dashdot", c='black')
plt.plot(fpr[3], tpr[3], label=f'PLCancer', linestyle = "dotted", c='black')
plt.plot(fpr['micro'], tpr['micro'], label='micro', lw=3, c='gray')
plt.legend()
plt.savefig('/home/imedix/amed_liver/noaug/efficientnetB0_rsf_batch32_liver_noaug_roc_curve4_micro.png')
#plt.show()

print('BLTumor_AUC:',auc(fpr[0], tpr[0]))
print('LCyst_AUC:',auc(fpr[1], tpr[1]))
print('MLCancer_AUC:',auc(fpr[2], tpr[2]))
print('PLCancer_AUC:',auc(fpr[3], tpr[3]))
print('Micro_AUC:',auc(fpr['micro'], tpr['micro']))
print('Macro_AUC:',auc(fpr['macro'], tpr['macro']))

#cf_matrix = cf_matrix.drop(cf_matrix.columns[[0]], axis=0)

sns.heatmap(cf_matrix, square=True, cbar=True, annot=True, cmap='Blues', fmt="d")
plt.yticks(rotation=0)
plt.xlabel("Predicted label", fontsize=13, rotation=0)
plt.ylabel("True label", fontsize=13)
#ax.set_ylim(len(cf_matrix), 0)
plt.savefig('efficientnetB0_rsf_batch32_liver_Heatmap_20220423_final_rsf.png')


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
