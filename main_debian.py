import numpy as np
from keras import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import chunks, queue_wrapper
import h5py
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt


def load_data(path):
    with h5py.File(path, 'r') as f:
        a_group_key = list(f.keys())[0]
        raw_data = list(f[a_group_key].items())
        bytes_columns = raw_data[0][1][()]
        columns = [i.decode('utf-8') for i in bytes_columns]
        data = raw_data[-1][1][()]
        return [columns, data] # 我在这个地方改的切片，完整代码就不考虑切片了吧

# 用来定义窗口的函数 lookback是滑动窗口大小 我们用的1000 lookahead是预测个数 我们是1 也就是1个1个的滑
def time_window(data, columns, sub_list, lookback, lookahead):
    X_tmp = np.zeros(shape=(1, lookback, len(columns)))
    y_tmp = np.zeros(shape=(1, lookahead, 2))  # 一个bid 一个ask

    for k in sub_list:
        X_window = np.array([data[k + j, :] for j in range(0, lookback)])
        X_tmp = np.concatenate((X_window.reshape(1, lookback, len(columns)), X_tmp), axis=0)
        del X_window
        y_window = np.array([data[k + lookback + z, [0, 2]] for z in range(0, lookahead)])
        y_tmp = np.concatenate((y_window.reshape(1, lookahead, 2), y_tmp), axis=0)
        del y_window

    return [X_tmp[:-1, ], y_tmp[:-1, ]] # 为了直接把数据集做成array，用了个空array垫底接concatenate，最后把它剃掉了


def organize_data(data, lookback, lookahead):

    PROC_COUNT = 24 # 多进程核数 我看了一下似乎可以放到config里面去 你看看能不能操作一下

    process_list = []
    results_list = []
    queue = Queue()

    # 给每个核心分配要切割的时间段，按时间顺序，调用了一个自定义的chunks
    _range = list(range(0, len(data)-lookback-lookahead, lookahead))
    _list = list(chunks(_range, int(len(_range)/PROC_COUNT)+1))

    X_tmp = np.zeros(shape=(1, lookback, len(columns)))
    y_tmp = np.zeros(shape=(1, lookahead, 2))  # 2 is the number of label columns
    ordered_list = [None] * PROC_COUNT

    for i in range(PROC_COUNT):
        p = Process(target=queue_wrapper, args=[queue, time_window, i, data, columns, _list[i], lookback, lookahead]) # 调用了一个自定义的queue_wrapper 就是我把每一次窗口划出来的array输出到queue里面
        process_list.append(p)
        p.start()

    for i in range(PROC_COUNT):
        results_list.append(queue.get())

    for p in process_list:
        p.join()

    for item in results_list:
        ordered_list[item[1]] = item[0]

    for item in ordered_list: # [[X_tmp, y_tmp], [X_tmp, t_tmp], ...] follow the timestamp order 调整一下时间 因为多进程分出来不一定有先后顺序 这一步把所有核心细分的array调整到正确的顺序
        X_tmp = np.concatenate((item[0], X_tmp), axis=0)
        y_tmp = np.concatenate((item[1], y_tmp), axis=0)

    return [X_tmp[:-1, ], y_tmp[:-1, ]]

# 分三个数据集
def split_data(X, y, validation_size, test_size):
    # X, y = np.array(X), np.array(y)
    X_test, y_test = X[len(X)-test_size:, ], y[len(y)-test_size:, ]
    X_val, y_val = X[len(X)-test_size-validation_size:len(X)-test_size, ], \
                   y[len(y)-test_size-validation_size:len(y)-test_size, ]
    X_train, y_train = X[:len(X)-test_size-validation_size, ], y[:len(y)-test_size-validation_size, ]
    return [X_train, y_train, X_val, y_val, X_test, y_test]


if __name__ == '__main__':

    LOOKBACK = 1000 # 这些常数看能不能放到config里面
    LOOKAHEAD = 1

    file_path = r'TSLA0107to0108.h5'
    columns, data = load_data(file_path)
    X, y = organize_data(data, LOOKBACK, LOOKAHEAD)
    with h5py.File('timeseriesdataset.hdf5', 'w') as f:
        x_dset = f.create_dataset('processed_data', (X.shape[0], X.shape[1], X.shape[2]), dtype='float64', data=X)
        y_dset = f.create_dataset('processed_label', (y.shape[0], y.shape[1], y.shape[2]), dtype='float64', data=y)
    # validation_size = test_size = int(.2 * len(X)) 这个是6-2-2 在数据集足够大合理的预期里用这个最好
    test_size = int(.5 * len(X))
    validation_size = int(.2 * .5 * len(X)) # 这俩是当时用两天数据想训练一天测试一天给的长度 效果比较差
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, validation_size, test_size)
    del X, y # 如果内存够大上面的那些del也可以不要 就想快一点进行垃圾回收

    model = Sequential()
    model.add(LSTM(units=32, input_shape=(LOOKBACK, len(columns)), return_sequences=False))
    # model.add(Dense(units=64, activation='relu')) 本来是打算按这个设计的 整理代码可以放进去 但为了计算快点少点参数我拿掉了
    model.add(Dense(units=2))

    adam = Adam(learning_rate=0.002)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
    # 保存最好的模型
    check_pointer = ModelCheckpoint(filepath='one_dense_weights_{epoch:03d}-{val_loss:.4f}.h5', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_train, y_train.reshape((-1, 2)), epochs=200, batch_size=32,
                        validation_data=(X_val, y_val.reshape((-1, 2))), callbacks=[check_pointer, early_stop])

    loss, acc = model.evaluate(X_test, y_test.reshape((-1, 2)))
    print('loss: ', loss)
    print('accuracy: ', acc)


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