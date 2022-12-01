import sys
print(sys.version)
print(sys.path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf

import os, cv2, zipfile, io, re, glob
from PIL import Image
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
#from keras.applications.vgg16 import VGG16
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from tensorflow.keras.applications.resnet import SeResNeXt50from keras.models import Model, load_model
# for keras
import keras
from classification_models.keras import Classifiers

# for tensorflow keras
from classification_models.tfkeras import Classifiers

from keras.layers import Input, Flatten, Dense, Activation, Dropout
from keras.layers.core import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

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
img_dirs = os.listdir(dirname + "/hikakudata/Train")
img_dirs.sort()
print(img_dirs)
classes = img_dirs
#classes = ['cyst','hcc','hemangioma','meta']

num_classes = len(classes)

del img_dirs

# 画像サイズ
image_size = 256
#image_size = 224

# 画像を取得し、配列に変換
def im2array(path):
    X = []
    y = []
    class_num = 0

    for class_name in classes:
        if class_num == num_classes : break
        imgfiles = glob.glob(path + class_name + "/*.bmp")
        for imgfile in imgfiles:
            # ZIPから画像読み込み
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

#trainデータ取得
X_train, y_train = im2array("hikakudata/Train/")

X_train = X_train.astype('float32')

# 正規化
X_train /= 255
#X_test /= 255

# one-hot 変換
y_train = to_categorical(y_train, num_classes = num_classes)

#各種関数定義
# モデル学習
def model_fit():
    hist = model.fit_generator(
        datagen.flow(X_train, y_train, batch_size = 32),
        steps_per_epoch = X_train.shape[0] // 32,
        epochs = 50,
        validation_data = (X_val, y_val),
        callbacks = [early_stopping, reduce_lr],
        shuffle = True,
        verbose = 1
    )
    return hist



def save_history(hist, result_file):
    loss_v = hist.history['loss']
    acc_v = hist.history['acc']
    val_loss_v = hist.history['val_loss']
    val_acc_v = hist.history['val_acc']
    nb_epoch = len(acc_v)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss_v[i], acc_v[i], val_loss_v[i], val_acc_v[i]))



#def model_save(model_name):
#    model.save(model_dir + 'model_' + model_name + '.hdf5')

    # optimizerのない軽量モデルを保存（学習や評価不可だが、予測は可能）
#    model.save(model_dir + 'model_' + model_name + '-opt.hdf5', include_optimizer = False)
def model_evaluate():
   #score = model.evaluate(X_test, y_test, verbose = 1)
    score = model.evaluate(X_val, y_val, verbose = 1)
    print("evaluate loss: {[0]:.4f}".format(score))
    print("evaluate acc: {[1]:.1%}".format(score))
    dt_now = datetime.datetime.now()
    fileobj = open("SeResNet50_all_256_liver_noaug_20210729_rsf_batch32_size224.txt", "w")
#sample.txtを書き出しモードで開きます。
    fileobj.write('{0:%Y_%m_%d_%H_%M_%S}'.format(dt_now) + "evaluate loss: {[0]:.4f}".format(score) + ' ' + "evaluate acc: {[1]:.1%}".format(score))
    fileobj.close()
    return score


# 学習曲線をプロット
def learning_plot(title):
    plt.figure(figsize = (18,6))

    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(hist.history["acc"], label = "acc", marker = "o")
    plt.plot(hist.history["val_acc"], label = "val_acc", marker = "o")
    #plt.yticks(np.arange())
    #plt.xticks(np.arange())
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend(loc = "best")
    plt.grid(color = 'gray', alpha = 0.2)

    # loss
    plt.subplot(1, 2, 2)
    plt.plot(hist.history["loss"], label = "loss", marker = "o")
    plt.plot(hist.history["val_loss"], label = "val_loss", marker = "o")
    #plt.yticks(np.arange())
    #plt.xticks(np.arange())
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend(loc = "best")
    plt.grid(color = 'gray', alpha = 0.2)

    plt.show()

#複数のGPUで動かす
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    models = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(kf.split(X_train,y_train)):

        X_tra=X_train[train_index]
        y_tra=y_train[train_index]
        X_val=X_train[val_index]
        y_val=y_train[val_index]

        #Data Augmentation
        datagen = ImageDataGenerator(
            featurewise_center = False,
            samplewise_center = False,
            featurewise_std_normalization = False,
            samplewise_std_normalization = False,
            zca_whitening = False,
            rotation_range = 0,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            horizontal_flip = True,
            vertical_flip = False
        )

        #Callback
        # EarlyStopping
        early_stopping = EarlyStopping(
            monitor = 'val_loss',
            patience = 10,
            verbose = 1
        )

        # ModelCheckpoint
        weights_dir = './weights/'
        if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
        model_checkpoint = ModelCheckpoint(
            weights_dir + "val_loss{val_loss:.3f}_SeResNeXt50_all_256_codid_noaug_20210729_rsf_batch32_size224.hdf5",
            monitor = 'val_loss',
            verbose = 1,
            save_best_only = True,
            save_weights_only = True,
            period = 3
        )

        # reduce learning rate
        reduce_lr = ReduceLROnPlateau(
            monitor = 'val_loss',
            factor = 0.1,
            patience = 3,
            verbose = 1
        )

        # log for TensorBoard
        logging = TensorBoard(log_dir = "log/")

        #結果テキスト保存
        nb_epoch = 50

        result_dir = 'results'
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        # モデル保存
        model_dir = './model/'
        weights_dir = './weights/'
        if os.path.exists(model_dir) == False : os.mkdir(model_dir)

        #SeResNeXt50
        SeResNeXt50, preprocess_input = Classifiers.get('seresnext50')
        base_model = SeResNeXt50(include_top = False, input_shape=(224, 224, 3), weights='imagenet')
        '''
        base_model = SeResNeXt50(
            include_top = False,
            weights = "imagenet",
            input_shape = None
        )
        '''

        # 全結合層の新規構築
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1024, activation = 'relu')(x)
        predictions = keras.layers.Dense(num_classes, activation = 'softmax')(x)

        # ネットワーク定義
        model = keras.models.Model(inputs = base_model.input, outputs = predictions)
        print("{}層".format(len(model.layers)))


        # layer.trainableの設定後にcompile
        model.compile(
            optimizer = Adam(),
            loss = 'categorical_crossentropy',
            metrics = ['acc']
        )

        hist=model_fit()

        model_evaluate()
        print('Fold:', {fold} )
        models.append(model)
    

# モデル評価

def model_save(model_name):
    model_json = model.to_json()
    with open(model_dir +"liver_256_noaug" + model_name + "_all_20210729_rsf_batch32_size224.json", "w") as json_file:
     json_file.write(model_json)

    model.save_weights(weights_dir + 'liver_256_noaug' + model_name + '_all_20210729_rsf_batch32_size224.h5')
    # optimizerのない軽量モデルを保存（学習や評価不可だが、予測は可能）

model_save("SeResNeXt50")
save_history(hist, os.path.join(result_dir, 'history_SeResNeXt50_all_256_liver_noaug_20210729_all_rsf_batch32_size224.txt'))
learning_plot("SeResNeXt50_all_liver_256_noaug_20210729_rsf_batch32_size224")
