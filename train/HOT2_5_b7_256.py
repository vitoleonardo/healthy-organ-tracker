#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import gc
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.notebook import tqdm
from datetime import datetime
import json,itertools
from typing import Optional
from glob import glob

from tensorflow import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import concatenate
from keras.losses import binary_crossentropy
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import load_model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D

from unet import build_unet

print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

INPUT = (256,256,3)
HEIGHT = INPUT[0]
WIDTH = INPUT[1]
name = "efficientnet-b7_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
backbone = 'efficientnetb7'

BATCH_SIZE = 16
EPOCHS=50
N_SPLITS=5
SELECTED_FOLD=1 # 1..5
lr = 5e-4

# For trianing purposes on HPC we will use a preconfigured fold using 
#skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
#for fold, (_, val_idx) in enumerate(skf.split(X=df_train, y=df_train['count'],groups =df_train['case']), 1):
#    df_train.loc[val_idx, 'fold'] = fold
USE_FOLD_CSV=True

# If true, loads pretrained model
DEBUG = False

BASE_PATH = 'input/uw-madison-gi-tract-image-segmentation/'
TRAIN_DIR =  BASE_PATH + 'train'
TRAIN_CSV =  BASE_PATH + 'train.csv'

df = pd.read_csv(TRAIN_CSV)
df.rename(columns = {'class':'class_name'}, inplace = True)
#--------------------------------------------------------------------------
df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
df["slice"] = df["id"].apply(lambda x: x.split("_")[3])
#--------------------------------------------------------------------------
all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
x = all_train_images[0].rsplit("/", 4)[0] ## ../input/uw-madison-gi-tract-image-segmentation/train

path_partial_list = []
for i in range(0, df.shape[0]):
    path_partial_list.append(os.path.join(x,
                          "case"+str(df["case"].values[i]),
                          "case"+str(df["case"].values[i])+"_"+ "day"+str(df["day"].values[i]),
                          "scans",
                          "slice_"+str(df["slice"].values[i])))
df["path_partial"] = path_partial_list
#--------------------------------------------------------------------------
path_partial_list = []
for i in range(0, len(all_train_images)):
    path_partial_list.append(str(all_train_images[i].rsplit("_",4)[0]))

tmp_df = pd.DataFrame()
tmp_df['path_partial'] = path_partial_list
tmp_df['path'] = all_train_images

#--------------------------------------------------------------------------
df = df.merge(tmp_df, on="path_partial").drop(columns=["path_partial"])
#--------------------------------------------------------------------------
df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_",4)[1]))
df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_",4)[2]))
#--------------------------------------------------------------------------
del x,path_partial_list,tmp_df
#--------------------------------------------------------------------------


# RESTRUCTURE  DATAFRAME
df_train = pd.DataFrame({'id':df['id'][::3]})

df_train['large_bowel'] = df['segmentation'][::3].values
df_train['small_bowel'] = df['segmentation'][1::3].values
df_train['stomach'] = df['segmentation'][2::3].values

df_train['path'] = df['path'][::3].values
df_train['case'] = df['case'][::3].values
df_train['day'] = df['day'][::3].values
df_train['slice'] = df['slice'][::3].values
df_train['width'] = df['width'][::3].values
df_train['height'] = df['height'][::3].values


df_train.reset_index(inplace=True,drop=True)
df_train.fillna('',inplace=True);
df_train['count'] = np.sum(df_train.iloc[:,1:4]!='',axis=1).values

# Remove faulty cases
fault1 = 'case7_day0'
fault2 = 'case81_day30'
df_train = df_train[~df_train['id'].str.contains(fault1) & ~df_train['id'].str.contains(fault2)].reset_index(drop=True)

# HELPER FUNCTIONS
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)


def build_masks(labels,input_shape, colors=True):
    height, width = input_shape
    if colors:
        mask = np.zeros((height, width, 3))
        for label in labels:
            mask += rle_decode(label, shape=(height,width , 3), color=np.random.rand(3))
    else:
        mask = np.zeros((height, width, 1))
        for label in labels:
            mask += rle_decode(label, shape=(height, width, 1))
    mask = mask.clip(0, 1)
    return mask


# Get Generator
class DataGenerator(tf.keras.utils.Sequence):
    # einfach nur n konstruktor
    def __init__(self, df, batch_size = BATCH_SIZE, width=WIDTH, height=HEIGHT, subset="train", shuffle=False):
        super().__init__()
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.on_epoch_end()

    # ohne self = static
    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        X = np.empty((self.batch_size,self.width,self.height,3))
        y = np.empty((self.batch_size,self.width,self.height,3))
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i,img_path in enumerate(self.df['path'].iloc[indexes]):
            w=self.df['width'].iloc[indexes[i]]
            h=self.df['height'].iloc[indexes[i]]
            img = self.__load_grayscale(img_path)
            X[i,] =img
            if self.subset == 'train':
                for k,j in zip([0,1,2],["large_bowel","small_bowel","stomach"]):
                    rles = self.df[j].iloc[indexes[i]]
                    masks = rle_decode(rles, shape=(h, w, 1))
                    masks = cv2.resize(masks, (self.width, self.height))
                    y[i,:,:,k] = masks
        if self.subset == 'train': return X, y
        else: return X


    # lädt bild und resized es
    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        dsize = (self.width, self.height)
        img = cv2.resize(img, dsize)
        img = img.astype(np.float32) / 255.
        # img = np.expand_dims(img, axis=-1)
        return img


# Cross Validation
#from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

#skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
#for fold, (_, val_idx) in enumerate(skf.split(X=df_train, y=df_train['count'],groups =df_train['case']), 1):
#    df_train.loc[val_idx, 'fold'] = fold

# Cross Validation; Import Index from CSV bc function is not available in module
if USE_FOLD_CSV:
    df_train['fold'] = pd.read_csv('fold.csv')['fold']

# Loss
from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(tf.cast(y_true, tf.float32), y_pred) + 0.5 * dice_loss(tf.cast(y_true, tf.float32), y_pred)


from segmentation_models import Unet, FPN
from segmentation_models.utils import set_trainable

input_weights = 'input/' + name

print("### Debug ###")
print("### Loading weights from + " name)

model = Unet(backbone,input_shape=INPUT, classes=3, activation='sigmoid',encoder_weights=input_weights)
opt = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=opt, loss=bce_dice_loss,metrics=[dice_coef,iou_coef]) #binary_crossentropy

from keras.callbacks import Callback, ModelCheckpoint

for i in range(1,5):
    MODEL_NAME = name + str(INPUT)+ "_BATCH_" + str(BATCH_SIZE) + "_EPOCHS_" + str(EPOCHS) + "_FOLD_" + str(i) + 'lr_' + str(lr)+ ".h5"
    log_dir = "logs/fit/"+ MODEL_NAME + datetime.now().strftime("%Y%m%d-%H%M%S") + str(i)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    callbacks = [
         tf.keras.callbacks.TensorBoard(
             log_dir=log_dir,
             histogram_freq=1),
         ModelCheckpoint(
             log_dir + "/model.h5",
             monitor='val_loss',
             verbose=0,
             save_best_only=True,
             save_weights_only=False,
             mode='auto'),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=8,
            verbose=0,
            min_delta=0.0001)
    ]

    train_ids = df_train[df_train["fold"]!=i].index
    valid_ids = df_train[df_train["fold"]==i].index
    
    train_generator = DataGenerator(df_train.loc[train_ids],shuffle=True)
    val_generator = DataGenerator(df_train.loc[valid_ids])

    history = model.fit(
    train_generator,
    validation_data=val_generator,
    callbacks=[callbacks],
    use_multiprocessing=False, 
    workers=4,
    epochs=EPOCHS)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(log_dir + '/historyFold' + str(i) + '.csv')





