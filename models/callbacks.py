import tensorflow as tf

def baseline_callback(model_name):
    file_path = f'{model_name}.h5'
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, verbose=0, save_best_only=True, save_weights_only=True)
    callbacks = [checkpointer]
    return callbacks