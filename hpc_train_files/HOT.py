#!/usr/bin/env python

"""Usage: HOT.py <backbone> <dim> <batch> <epochs> <semi3d_data> <remove_faulty_cases> <use_crop_data> <selected_fold>

Backbones available: 
        'efficientnetb0'
        'efficientnetb1'
        'efficientnetb2'
        'efficientnetb3'
        'efficientnetb4'
        'efficientnetb5'
        'efficientnetb6'
        'efficientnetb7'
        'inceptionresnetv2'
        'inceptionv3'
        'resnet50'
        'resnext50'
        'resnext101'
        'seresnext50'
        'seresnext101'
        'densenet121'
        'densenet201'

Dim should be divisible by 32
Batchsize standard is 16, increase if you have a lot of memory
Epochs standard is 50, increase if you have a lot of time
Semi3d_data is a boolean, True if you want to train with 2.5D data
Remove_faulty_cases is a boolean, True if you want to remove faulty cases from the training set

Options:
  -h --help     show this
  -v            verbose mode
"""
from docopt import docopt

import warnings
import sys
import os
import subprocess
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
from datapreparation import extract_metadata, remove_faulties

print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

def main(backbone, dim, batch, epochs, semi3d_data, remove_faulty_cases, use_crop_data, selected_fold):
    cfg = CFG(
        backbone            = backbone,
        img_dims            = (dim,dim,3),
        model               = 'UNet',
        batch_size          = batch, 
        epochs              = epochs, 
        kaggle              = False, 
        use_fold_csv        = True,
        semi3d_data         = semi3d_data,
        remove_faulty_cases = remove_faulty_cases,
        use_crop_data       = use_crop_data,
        selected_fold       = selected_fold)

    print("Starting training with config: ", str(cfg))

    df_train = pd.read_csv("df_train.csv", index_col=[0])
    df_train.fillna('',inplace=True); 

    print(df_train.info(verbose=True))

    if remove_faulty_cases:
        df_train = remove_faulties(df_train)
        print("Removed faulty cases")

    # Cross Validation; Import Index from CSV bc function is not available in HPC module
    if not cfg.use_fold_csv:
        from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

        skf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        for fold, (_, val_idx) in enumerate(skf.split(X=df_train, y=df_train['count'], groups=df_train['case']), 1):
            df_train.loc[val_idx, 'fold'] = fold
    else:
        df_train['fold'] = pd.read_csv('fold.csv')['fold']

    weights = cfg.get_encoder_weights_path(cfg.backbone)
    print("Loading weights from: ", weights)

    install()

    from segmentation_models import Unet, FPN
    from segmentation_models.utils import set_trainable
    from keras.callbacks import Callback, ModelCheckpoint

    model   = Unet(cfg.backbone,input_shape=cfg.img_dims, classes=3, activation='sigmoid', encoder_weights=cfg.encoder_weights_path[cfg.backbone])
    opt     = tf.keras.optimizers.Adam(learning_rate=cfg.lr)
        
    model.compile(optimizer=opt, loss=bce_dice_loss,metrics=[dice_coef,iou_coef])

    # Train on selected fold
    i = cfg.selected_fold

    model_full_name = f"{cfg.model}_BB_{cfg.backbone}_DIM_{cfg.img_dims}_SEMI3D_{cfg.semi3d_data}_CROPDATA_{cfg.use_crop_data}_FAULTIES_{cfg.remove_faulty_cases}_BATCH_{cfg.batch_size}_EPOCHS_{cfg.epochs}_FOLD_{i}.h5"
    log_dir = f'logs/best/{model_full_name}_{datetime.now().strftime("%d%m%Y-%H%M")}_FOLD_{i}'

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
    
    train_generator = DataGenerator(df_train.loc[train_ids],batch_size=cfg.batch_size, height=cfg.height, width=cfg.width,shuffle=True, semi3d_data=cfg.semi3d_data, use_crop_data=cfg.use_crop_data)
    val_generator = DataGenerator(df_train.loc[valid_ids], height=cfg.height, width=cfg.width, semi3d_data=cfg.semi3d_data)

    print(f"Starting training {model_full_name}")

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

def install():
    subprocess.call(['pip', 'install', 'segmentation_models'], stdout=open(os.devnull, 'wb'))

if __name__ == '__main__':
    args = docopt(__doc__, version='HOT 0.1')
    print("Starting training. Arguments: \n" + str(args))

    try:
        backbone = args['<backbone>']
        dim = int(args['<dim>'])
        batch = int(args['<batch>'])
        epochs = int(args['<epochs>'])
        semi3d_data = json.loads(args['<semi3d_data>'].lower())
        remove_faulty_cases = json.loads(args['<remove_faulty_cases>'].lower())
        use_crop_data = json.loads(args['<use_crop_data>'].lower())
        selected_fold = int(args['<selected_fold>'])
        main(backbone, dim, batch, epochs, semi3d_data, remove_faulty_cases, use_crop_data, selected_fold)
    except Exception as e:
        print(e)
        print("Error: Check arguments")
        sys.exit(1)
