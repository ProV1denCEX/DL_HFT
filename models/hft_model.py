import h5py

from base.base_model import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


class LSTM_HFT(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.lookback = config['LOOKBACK']
        self.num_col = config['num_col']
        self.build_model()

        self.history = None

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=32, input_shape=(self.lookback, self.num_col)))
        self.model.add(Dense(units=64, activation='relu'))

        self.model.compile(optimizer=Adam(learning_rate=0.002), loss='mean_squared_error', metrics=['accuracy'])

    def fit(self, X_train, y_train, X_val, y_val):

        check_pointer = ModelCheckpoint(filepath='one_dense_weights_{epoch:03d}-{val_loss:.4f}.h5', verbose=1,
                                        save_best_only=True)

        early_stop = EarlyStopping(monitor='val_loss', patience=10)

        self.history = self.model.fit(X_train, y_train.reshape((-1, 2)), epochs=200, batch_size=32,
                            validation_data=(X_val, y_val.reshape((-1, 2))), callbacks=[check_pointer, early_stop])

    def evaluate(self, X_test, y_test):
        loss, acc = self.model.evaluate(X_test, y_test.reshape((-1, 2)))

        print('loss: ', loss)
        print('accuracy: ', acc)

    def plot_model(self):
        history_dic = self.history.history
        loss_values = history_dic['loss']
        validation_loss_values = history_dic['val_loss']
        epochs = range(1, len(loss_values) + 1)

        plt.figure(figsize=(15, 8))
        plt.plot(epochs, loss_values, 'bo', label="Training loss")
        plt.plot(epochs, validation_loss_values, 'bo', label="validation loss")
        plt.title('Trainig and validation loss for early stopping model')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss.png')

    def predict(self, X):
        # 存预测值
        results = self.model.predict(X)
        with h5py.File('predictions.hdf5', 'w') as f:
            results_dset = f.create_dataset('predictions', (results.shape[0], results.shape[1]), dtype='float64',
                                            data=results)


if __name__ == '__main__':
    model = LSTM_HFT({"LOOKBACK": 1000, 'num_col': 12})
