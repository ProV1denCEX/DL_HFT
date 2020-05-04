"""
因为是当脚本上传跑的 所以基本上和主程序一样的 效果是单独提取预测数据集 然后加载模型预测
当时没保存预测数据先写了这个脚本 因为分数据集要时间就单独再加了个读数据之后预测的脚本 我现在合到一起了 就加了个保存预测值的部分
"""

import numpy as np
from keras.models import load_model
import h5py
from multiprocessing import Process, Queue


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def queue_wrapper(queue, f, index, *args):
    queue.put((f(*args), index))


def load_data(path):
    with h5py.File(path, 'r') as f:
        a_group_key = list(f.keys())[0]
        raw_data = list(f[a_group_key].items())
        bytes_columns = raw_data[0][1][()]
        columns = [i.decode('utf-8') for i in bytes_columns]
        data = raw_data[-1][1][()]
        return [columns, data[150000:200000, ]] # [:20000, ]


def time_window(data, columns, sub_list, lookback, lookahead):
    X_tmp = np.zeros(shape=(1, lookback, len(columns)))
    y_tmp = np.zeros(shape=(1, lookahead, 2))  # 2 is the number of label columns 可删 不想给下面赋太多变量


    for k in sub_list:
        X_window = np.array([data[k + j, :] for j in range(0, lookback)])
        X_tmp = np.concatenate((X_window.reshape(1, lookback, len(columns)), X_tmp), axis=0)
        del X_window
        y_window = np.array([data[k + lookback + z, [0, 2]] for z in range(0, lookahead)])
        y_tmp = np.concatenate((y_window.reshape(1, lookahead, 2), y_tmp), axis=0)
        del y_window

    return [X_tmp[:-1, ], y_tmp[:-1, ]]


def organize_data(data, lookback, lookahead):

    PROC_COUNT = 24

    process_list = []
    results_list = []
    queue = Queue()
    _range = list(range(0, len(data)-lookback-lookahead, lookahead))
    _list = list(chunks(_range, int(len(_range)/PROC_COUNT)+1))

    X_tmp = np.zeros(shape=(1, lookback, len(columns)))
    y_tmp = np.zeros(shape=(1, lookahead, 2))  # 2 is the number of label columns
    ordered_list = [None] * PROC_COUNT

    for i in range(PROC_COUNT):
        p = Process(target=queue_wrapper, args=[queue, time_window, i, data, columns, _list[i], lookback, lookahead])
        process_list.append(p)
        p.start()

    for i in range(PROC_COUNT):
        results_list.append(queue.get())

    for p in process_list:
        p.join()

    for item in results_list:
        ordered_list[item[1]] = item[0]

    for item in ordered_list: # [[X_tmp, y_tmp], [X_tmp, t_tmp], ...] follow the timestamp order
        X_tmp = np.concatenate((item[0], X_tmp), axis=0)
        y_tmp = np.concatenate((item[1], y_tmp), axis=0)

    return [X_tmp[:-1, ], y_tmp[:-1, ]]




if __name__ == '__main__':

    LOOKBACK = 1000
    LOOKAHEAD = 1

    file_path = r'TSLA0107to0108.h5'
    columns, data = load_data(file_path)
    X_test, y_test = organize_data(data, LOOKBACK, LOOKAHEAD)
    with h5py.File('test_set.hdf5', 'w') as f:
        x_dset = f.create_dataset('processed_data', (X_test.shape[0], X_test.shape[1], X_test.shape[2]), dtype='float64', data=X_test)
        y_dset = f.create_dataset('processed_label', (y_test.shape[0], y_test.shape[1], y_test.shape[2]), dtype='float64', data=y_test)

    model = load_model('one_dense_weights_008-0.7162.h5')

    loss, acc = model.evaluate(X_test, y_test.reshape((-1, 2)))
    print('loss: ', loss)
    print('accuracy: ', acc)

    # 存预测值
    results = model.predict(data)
    with h5py.File('predictions.hdf5', 'w') as f:
        results_dset = f.create_dataset('predictions', (results.shape[0], results.shape[1]), dtype='float64', data=results)