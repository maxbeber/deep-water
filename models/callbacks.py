from tensorflow.keras import callbacks


def baseline_callback(model_name):
    file_path = f'{model_name}.h5'
    callback = callbacks.ModelCheckpoint(\
        filepath=file_path,
        save_best_only=True,
        save_weights_only=True,
        verbose=0)
    return [callback]


def build_callbacks(parameters):
    model_name = parameters['model_name']
    is_logger_enabled = parameters['enable_csv_logger']
    checkpoint = model_checkpoint(model_name)
    logger = csv_logger(model_name)
    early_stop = early_stopping(parameters)
    reduce_learning_rate = reduce_learning_rate_on_plateau(parameters)
    callbacks = checkpoint + early_stop + reduce_learning_rate
    if is_logger_enabled:
        callbacks = callbacks + logger       
    return callbacks


def csv_logger(model_name):
    file_path = f'training_{model_name}.log'
    callback = callbacks.CSVLogger(file_path, separator=',', append=False)
    return [callback]


def early_stopping(parameters):
    mode = parameters['early_stopping']['mode']
    monitor = parameters['early_stopping']['monitor']
    patience = parameters['early_stopping']['patience']
    callback = callbacks.EarlyStopping(mode=mode, monitor=monitor, patience=patience)
    return [callback]


def model_checkpoint(model_name):
    file_path = f'{model_name}.h5'
    callback = callbacks.ModelCheckpoint(\
        filepath=file_path,
        save_best_only=True,
        save_weights_only=True,
        verbose=0)
    return [callback]


def reduce_learning_rate_on_plateau(parameters):
    cooldown = parameters['reduce_lr_on_plateau']['cooldown']
    factor = parameters['reduce_lr_on_plateau']['factor']
    min_delta = parameters['reduce_lr_on_plateau']['min_delta']
    min_lr = parameters['reduce_lr_on_plateau']['min_lr']
    monitor = parameters['reduce_lr_on_plateau']['monitor']
    patience = parameters['reduce_lr_on_plateau']['patience']
    callback = callbacks.ReduceLROnPlateau(\
        cooldown=cooldown,
        factor=factor,
        min_delta=min_delta,
        min_lr=min_lr,
        mode='auto',
        monitor=monitor,
        patience=patience,
        verbose=1)
    return [callback]