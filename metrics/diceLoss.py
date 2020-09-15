import tensorflow as tf


def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    coeficient = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return coeficient


def dice_coef_loss(y_true, y_pred):
    loss = 1.0 - dice_coef(y_true, y_pred)
    return loss