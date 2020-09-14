import tensorflow as tf


def mean_iou(y_true, y_pred):
    """
    The intersection over union (IoU) or Jacard metric
    is used to evaluate the performance of a segmentation algorithm.

    Parameters
    ----------
    y_true : the true value
    y_pred : the estimated value

    Returns the computed metric as floating point 32 bytes.
    """
    threshold = 0.5
    yt0 = y_true[:, :, :, 0]
    yp0 = tf.keras.backend.cast(y_pred[:, :, :, 0] > threshold, 'float32')
    intersection = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1.0, tf.cast(intersection/union, 'float32'))
    return iou


def iou_loss(y_true, y_pred):
    loss = 1 - mean_iou(y_true, y_pred)
    return loss