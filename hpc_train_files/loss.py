# Loss
from typing import Callable
from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf

# zitieren
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

def dice_coef2(y_true, y_pred):
    """
    The function takes in two tensors, y_true and y_pred, and returns the dice coefficient.
    This is another implementation for evaluating trained models.
    
    :param y_true: Ground truth values. shape = [batch_size, height, width, 1]
    :param y_pred: The predicted output of the model
    :return: The dice coefficient.
    """
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

# Thanks to @maxvfischer https://github.com/keras-team/keras/issues/93954
def dice_coef_single_label(class_idx: int, name: str, epsilon=1e-6) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Extract single class to compute dice coef
        y_true_single_class = y_true[..., class_idx]
        y_pred_single_class = y_pred[..., class_idx]

        intersection = K.sum(K.abs(y_true_single_class * y_pred_single_class))
        return (2. * intersection) / (K.sum(K.square(y_true_single_class)) + K.sum(K.square(y_pred_single_class)) + epsilon)

    dice_coef.__name__ = f"dice_coef_{name}"  # Set name used to log metric
    return dice_coef

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
    > The function takes in two arguments, the true labels and the predicted labels, and returns the
    weighted sum of the binary crossentropy loss and the dice loss
    
    :param y_true: the ground truth mask
    :param y_pred: The predicted output of the model
    :return: The loss function is being returned.
    """
    return 0.6 * binary_crossentropy(tf.cast(y_true, tf.float32), y_pred) + 0.4 * dice_loss(tf.cast(y_true, tf.float32), y_pred)

