# Loss
from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1):
    """
    The dice coefficient is a measure of overlap between two samples
    
    :param y_true: Ground truth
    :param y_pred: The predicted output of the model
    :param smooth: A small constant to avoid division by zero, defaults to 1 (optional)
    :return: The dice coefficient.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1):
    """
    The function takes in two tensors, y_true and y_pred, and calculates the intersection over union
    (IoU) between them
    
    :param y_true: Ground truth mask
    :param y_pred: The predicted mask
    :param smooth: This is used to avoid division by zero, defaults to 1 (optional)
    :return: The mean of the IoU for each image in the batch.
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_loss(y_true, y_pred):
    """
    It takes the intersection of the two masks, divides by the sum of the two masks, and subtracts the
    result from 1
    
    :param y_true: the ground truth mask
    :param y_pred: The predicted output of the model
    :return: The dice loss function is being returned.
    """
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    """
    It takes the binary cross entropy loss and adds to it the dice loss with a factor of 0.5
    
    :param y_true: the ground truth mask
    :param y_pred: The predicted output of the model
    :return: The loss function is being returned.
    """
    return binary_crossentropy(tf.cast(y_true, tf.float32), y_pred) + 0.5 * dice_loss(tf.cast(y_true, tf.float32), y_pred)