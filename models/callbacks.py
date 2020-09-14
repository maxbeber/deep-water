from tensorflow.keras import callbacks


def baseline_callback(model_name):
    file_path = f'{model_name}.h5'
    callback = callbacks.ModelCheckpoint(filepath=file_path, verbose=0, save_best_only=True, save_weights_only=True)
    return [callback]


def build_callbacks(model_name, monitor, mode, patience, factor, min_delta, cooldown, min_learning_rate):
    checkpoint = model_checkpoint(model_name, mode, monitor)
    early_stop = early_stopping(mode, monitor, patience)
    reduce_learning_rate = reduce_learning_rate_on_plateau(factor, min_delta, min_learning_rate, monitor, patience)
    callbacks = checkpoint + early_stop + reduce_learning_rate
    return callbacks


def early_stopping(mode, monitor, patience):
    callback = callbacks.EarlyStopping(mode=mode, monitor=monitor, patience=patience)
    return [callback]


def model_checkpoint(model_name, mode, monitor):
    file_path = f'{model_name}.h5'
    callback = callbacks.ModelCheckpoint(\
        filepath=file_path,
        mode=mode,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=True,
        verbose=0)
    return [callback]


def reduce_learning_rate_on_plateau(factor, min_delta, min_learning_rate, monitor, patience):
    callback = callbacks.ReduceLROnPlateau(\
        factor=factor,
        min_delta=min_delta,
        min_lr=min_learning_rate,
        mode='auto',
        monitor=monitor,
        patience=patience,
        verbose=1)
    return [callback]