"""
这是当时发现运行的主程序突然不见了之后补的一个继续训练的脚本 reload_data和predict脚本里的一致
"""
import h5py
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
import matplotlib.pyplot as plt

def reload_data(path):
    with h5py.File(path, 'r') as f:
        data_group_key = list(f.keys())[0]
        label_group_key = list(f.keys())[1]
        data = np.array(f[data_group_key])
        label = np.array(f[label_group_key])
        return [data, label]

def split_data(X, y, validation_size, test_size):
    # X, y = np.array(X), np.array(y)
    X_test, y_test = X[len(X)-test_size:, ], y[len(y)-test_size:, ]
    X_val, y_val = X[len(X)-test_size-validation_size:len(X)-test_size, ], \
                   y[len(y)-test_size-validation_size:len(y)-test_size, ]
    X_train, y_train = X[:len(X)-test_size-validation_size, ], y[:len(y)-test_size-validation_size, ]
    return [X_train, y_train, X_val, y_val, X_test, y_test]


if __name__ == '__main__':
    file_path = r'timeseriesdataset.hdf5'
    data, label = reload_data(file_path)
    validation_size = test_size = int(.2 * len(data))
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(data, label, validation_size, test_size)
    model = load_model('one_dense_weights_008-0.7162.h5')

    check_pointer = ModelCheckpoint(filepath='one_dense_weights_continue_{epoch:03d}-{val_loss:.4f}.h5', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_train, y_train.reshape((-1, 2)), epochs=200, batch_size=32,
                        validation_data=(X_val, y_val.reshape((-1, 2))), callbacks=[check_pointer, early_stop])
    loss, acc = model.evaluate(X_test, y_test.reshape((-1, 2)))
    print('loss: ', loss)
    print('accuracy: ', acc)

    with open('train_history_dict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)         # 这里把history也存了下来

    history_dic = history.history
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