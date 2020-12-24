import os

class UnetBase:
    """
    Represents a base class for a deep neural network using an U-Net architecture.

    Parameters
    ----------
    model_name: the name of the model.
    image_size : the size of the image for the input layer.
    """
    def __init__(self):
        self.model = None


    def evaluate(self, x_test, y_test):
        metrics = self.model.evaluate(x_test, y_test, verbose=0)
        return metrics


    def get_model_summary(self):
        model_summary = self.model.summary()
        return model_summary
    

    def fit(self, train_generator, train_steps, validation_generator, validation_steps, epochs, callbacks):
        self.history = self.model.fit(train_generator,\
            steps_per_epoch=train_steps,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            epochs=epochs,
            verbose=1)


    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred

    
    def restore(self, model_path):
        file_exists = os.path.exists(model_path)
        if not file_exists:
            print(f'{model_path} could not be found.')
            return
        self.model.load_weights(model_path)


    def set_compiler(self, loss, metrics, optimizer):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)