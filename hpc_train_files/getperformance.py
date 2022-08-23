
#!/usr/bin/env python

"""Usage: getperformance.py <path>

Path to base dir where trained models are

This script evaluates trained models on the validation fold and saves a dataframe with 
every prediction mask rle encoded as well as the dice score of each prediction 

Options:
  -h --help     show this
  -v            verbose mode
"""
from docopt import docopt


import warnings
import pandas as pd
import sys
import re
from glob import glob
import cv2
import os
import numpy as np

from tensorflow import keras
import tensorflow as tf
from keras import backend as K
from keras.models import load_model

from loss import dice_coef, iou_coef, dice_coef2, bce_dice_loss
from utility import rle_encode, rle_decode
from dataloader import DataGenerator

warnings.filterwarnings("ignore")

class FixedDropout(keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

def get_meta_info_from_model_path(file):
    regexBone = 'None' if not re.search("BB_(\w+)_D",file) else re.search("BB_(\w+)_D",file).group(1) 
    regexDim = 'None' if not re.search("DIM_\((\d+),",file) else int(re.search("DIM_\((\d+),",file).group(1) )
    threeDim = 'None' if not re.search("SEMI3D_([aA-zZ]{4,5})_",file) else re.search("SEMI3D_([aA-zZ]{4,5})_",file).group(1) 
    regexCrop = 'None' if not re.search("CROPDATA_([aA-zZ]{4,5})_",file) else re.search("CROPDATA_([aA-zZ]{4,5})_",file).group(1) 
    regexBatch = 'None' if not re.search("BATCH_(\d+)_",file) else int(re.search("BATCH_(\d+)_",file).group(1) )
    regexEpochs = 'None' if not re.search("EPOCHS_(\d+)_",file) else int(re.search("EPOCHS_(\d+)_",file).group(1))
    regexFold = 'None' if not re.search("Fold_(\d).",file) else int(re.search("Fold_(\d).",file).group(1))

    if regexFold == 'None':
        if re.search("FOLD_(\d)",file):
            regexFold = int(re.search("FOLD_(\d)",file).group(1))
        else:
            regexFold = 3

    return regexBone, regexDim, threeDim, regexCrop, regexBatch, regexEpochs, regexFold

def evaluate_model( custom_objects, path="../tensorboard_logs/baselines_256_16_no_add_ons/UNet_BB_efficientnetb0_DIM_(256, 256, 3)_SEMI3D_False_CROPDATA_False_BATCH_16_EPOCHS_50_FOLD_3.h5_15082022-2107_FOLD_3/model.h5"):
    # Load the model
    print(f"Loading model from {path}")
    model = load_model(path, custom_objects=custom_objects)

    # Get metainfo from modelpath
    regexBone, regexDim, threeDim, _, regexBatch, _, regexFold = get_meta_info_from_model_path(path)
    
    # print regex parameters 
    print(f"Bone: {regexBone}")
    print(f"Dim: {regexDim}")
    print(f"ThreeDim: {threeDim}")
    print(f"Batch: {regexBatch}")
    print(f"Fold: {regexFold}")

    # Prepare validation batch
    df_train = pd.read_csv("df_train_with_mask_paths.csv")
    df_train.fillna('', inplace=True)

    # print df_train length
    print(f"df_train length: {len(df_train)}")

    df_train['fold'] = pd.read_csv('fold.csv')['fold']
    valid_ids = df_train[df_train["fold"]==regexFold].index
    df_val = df_train.loc[valid_ids].reset_index()

    print(f"df_val length: {len(df_val)}")

    val_generator = DataGenerator(df_val, height=int(regexDim), width=int(regexDim), semi3d_data=eval(threeDim), batch_size=int(regexBatch), shuffle=False, subset='test')
    num_batches = int(len(df_val)/regexBatch)

    print("Starting evaluation ... for batches: ", num_batches)

    for i in range(num_batches):
        LOGITS = model.predict(val_generator[i])     
        
        for j in range(int(regexBatch)):
            w = df_val.loc[i*int(regexBatch)+j,"width"]
            h = df_val.loc[i*int(regexBatch)+j,"height"]

            pred_img = cv2.resize(LOGITS[j], (w,h), interpolation=cv2.INTER_NEAREST)
            pred_img = (pred_img>0.5).astype(dtype='float32')    # classify

            mask_path = df_val.loc[i*int(regexBatch)+j,"multilabel_mask_path"]
            mask_img = cv2.imread(mask_path).astype(dtype='float32')

            for k,class_ in enumerate(["large_bowel","small_bowel","stomach"]):
                predicted_mask = class_ + "_predicted"
                class_ += "_dice_coef"

                nonzeros = np.count_nonzero(pred_img[:,:,k])
                nonzeros += np.count_nonzero(mask_img[:,:,k])

                if(nonzeros == 0):
                    df_val.loc[i*int(regexBatch)+j,class_] = np.nan
                    continue

                # attach the predicted mask to df_val

                dice_coef = dice_coef2(mask_img[:,:,k], pred_img[:,:,k])
                df_val.loc[i*int(regexBatch)+j,predicted_mask] = rle_encode(pred_img[:,:,k])
                df_val.loc[i*int(regexBatch)+j,class_] = float(dice_coef)


    df_val.to_csv(f"{path}_val_results.csv", index=False)

    return df_val

def main(path):

    custom_objects = custom_objects={
        'FixedDropout': FixedDropout,
        'dice_coef': dice_coef,
        'iou_coef': iou_coef,
        'bce_dice_loss': bce_dice_loss  
    }


    for model_path in glob(os.path.join(path, "**", "model.h5"), recursive=True):
        evaluate_model(custom_objects, model_path)




if __name__ == '__main__':
    args = docopt(__doc__, version='getperformance 0.0.1')
    print("Starting inference. Arguments: \n" + str(args))

    try:
        path = args['<path>']
        main(path)
    except Exception as e:
        print(e)
        print("Error: Check arguments")
        sys.exit(1)
