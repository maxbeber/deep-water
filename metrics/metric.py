import tensorflow as tf


def mean_iou(y_true, y_pred):
    yt0 = y_true[:, :, :, 0]
    yp0 = tf.keras.backend.cast(y_pred[:, :, :, 0] > 0.5, 'float32')
    intersection = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(intersection/union, 'float32'))
    return iou   