#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from datetime import datetime
import json,itertools

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

from config import CFG
from dataloader import DataGenerator
from utility import rle_encode, rle_decode, build_masks
from loss import dice_coef, iou_coef, dice_loss, bce_dice_loss
from datapreparation import extract_metadata

print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

cfg = CFG(
    img_dims            = (256,256,3),
    model               = 'UNet',
    batch_size          = 16, 
    epochs              = 50, 
    kaggle              = False, 
    use_fold_csv        = True,
    semi3d_data         = True,
    remove_faulty_cases = True)

df = pd.read_csv(cfg.train_csv)

DEBUG = False
if DEBUG:
    df = df.sample(n=90, random_state=cfg.seed)

# Extract metadata from image paths
df_train = extract_metadata(df, cfg.train_dir, remove_faulty_cases=True)

# Cross Validation; Import Index from CSV bc function is not available in HPC module
if not cfg.use_fold_csv:
    from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

    skf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    for fold, (_, val_idx) in enumerate(skf.split(X=df_train, y=df_train['count'], groups=df_train['case']), 1):
        df_train.loc[val_idx, 'fold'] = fold
else:
    df_train['fold'] = pd.read_csv('fold.csv')['fold']

from segmentation_models import Unet, FPN
from segmentation_models.utils import set_trainable

weights = cfg.get_encoder_weights_path(cfg.backbone)
print("Loading weights from: ", weights)

model = Unet(cfg.backbone,input_shape=cfg.img_dims, classes=3, activation='sigmoid', encoder_weights=cfg.encoder_weights_path[cfg.backbone])
opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr)
model.compile(optimizer=opt, loss=bce_dice_loss,metrics=[dice_coef,iou_coef])

# Train model on five folds
from keras.callbacks import Callback, ModelCheckpoint

for i in cfg.folds:
    model_full_name = cfg.model + str(cfg.img_dims)+ "_BATCH_" + str(cfg.batch_size) + "_EPOCHS_" + str(cfg.epochs) + "_FOLD_" + str(i) + '_lr_' + str(cfg.lr)+ ".h5"
    log_dir = "logs/"+ model_full_name + datetime.now().strftime("%Y%m%d-%H%M%S") + "_FOLD_" + str(i)

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
            patience=cfg.lr_patience,
            verbose=0,
            min_delta=0.0001)
    ]

    train_ids = df_train[df_train["fold"]!=i].index
    valid_ids = df_train[df_train["fold"]==i].index
    
    train_generator = DataGenerator(df_train.loc[train_ids],batch_size=cfg.batch_size, height=cfg.height, width=cfg.width,shuffle=True, semi3d_data=cfg.semi3d_data)
    val_generator = DataGenerator(df_train.loc[valid_ids], height=cfg.height, width=cfg.width, semi3d_data=cfg.semi3d_data)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        callbacks=[callbacks],
        use_multiprocessing=False, 
        workers=4,
        epochs=cfg.epochs)

    print("Finished training fold: " + str(i) + "... Continuing to next fold")

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(log_dir + '/historyFold' + str(i) + '.csv')