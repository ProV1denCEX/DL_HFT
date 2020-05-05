import h5py
import numpy as np

from multiprocessing import Process, Queue, cpu_count

from base.base_data_loader import BaseDataLoader
from utils.utils import chunks, queue_wrapper


class DataLoader(BaseDataLoader):
    def __init__(self, config=None):
        super().__init__(config)
        self.lookback = config['LOOKBACK']
        self.lookahead = config['LOOKAHEAD']

        self.columns = None
        self.data = None
        self.X = None
        self.y = None
        self.core = cpu_count()

    def load_data(self, path):
        with h5py.File(path, 'r') as f:
            a_group_key = list(f.keys())[0]
            raw_data = list(f[a_group_key].items())
            bytes_columns = raw_data[0][1][()]
            columns = [i.decode('utf-8') for i in bytes_columns]
            data = raw_data[-1][1][()]

            self.columns = columns
            self.data = data

    # 用来定义窗口的函数 lookback是滑动窗口大小 我们用的1000 lookahead是预测个数 我们是1 也就是1个1个的滑
    def time_window(self, sub_list, lookback, lookahead):
        X_tmp = np.zeros(shape=(1, lookback, len(self.columns)))
        y_tmp = np.zeros(shape=(1, lookahead, 2))  # 一个bid 一个ask

        for k in sub_list:
            X_window = np.array([self.data[k + j, :] for j in range(0, lookback)])
            X_tmp = np.concatenate((X_window.reshape(1, lookback, len(self.columns)), X_tmp), axis=0)
            del X_window
            y_window = np.array([self.data[k + lookback + z, [0, 2]] for z in range(0, lookahead)])
            y_tmp = np.concatenate((y_window.reshape(1, lookahead, 2), y_tmp), axis=0)
            del y_window

        return [X_tmp[:-1, ], y_tmp[:-1, ]]

    def organize_data(self):
        process_list = []
        results_list = []
        queue = Queue()

        # 给每个核心分配要切割的时间段，按时间顺序，调用了一个自定义的chunks
        _range = list(range(0, len(self.data) - self.lookback - self.lookahead, self.lookahead))
        _list = list(chunks(_range, int(len(_range) / self.core) + 1))

        X_tmp = np.zeros(shape=(1, self.lookback, len(self.columns)))
        y_tmp = np.zeros(shape=(1, self.lookahead, 2))
        ordered_list = [None] * self.core

        for i in range(self.core):
            p = Process(target=queue_wrapper, args=[queue,
                                                    self.time_window,
                                                    i,
                                                    _list[i],
                                                    self.lookback,
                                                    self.lookahead])
            process_list.append(p)
            p.start()

        for i in range(self.core):
            results_list.append(queue.get())

        for p in process_list:
            p.join()

        for item in results_list:
            ordered_list[item[1]] = item[0]

        # [[X_tmp, y_tmp], [X_tmp, t_tmp], ...] follow the timestamp order
        for item in ordered_list:
            X_tmp = np.concatenate((item[0], X_tmp), axis=0)
            y_tmp = np.concatenate((item[1], y_tmp), axis=0)

        self.X = X_tmp[:-1, ]
        self.y = y_tmp[:-1, ]

    # 分三个数据集
    def split_data(self, validation_size, test_size):
        # X, y = np.array(X), np.array(y)
        self.X_test, self.y_test = self.X[len(self.X) - test_size:, ], self.y[len(self.y) - test_size:, ]
        self.X_val, self.y_val = self.X[len(self.X) - test_size - validation_size:len(self.X) - test_size, ], \
                       self.y[len(self.y) - test_size - validation_size:len(self.y) - test_size, ]
        self.X_train, self.y_train = self.X[:len(self.X) - test_size - validation_size, ], self.y[:len(self.y) - test_size - validation_size, ]

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_val_data(self):
        return self.X_val, self.y_val


if __name__ == '__main__':
    data_loader = DataLoader({"LOOKBACK": 1000, "LOOKAHEAD": 1})
    data_loader.load_data("data/TSLA0107to0108.h5")
    data_loader.organize_data()

    with h5py.File('timeseriesdataset.hdf5', 'w') as f:
        f.create_dataset('processed_data', (data_loader.X.shape[0], data_loader.X.shape[1], data_loader.X.shape[2]), dtype='float64', data=data_loader.X)
        f.create_dataset('processed_label', (data_loader.y.shape[0], data_loader.y.shape[1], data_loader.y.shape[2]), dtype='float64', data=data_loader.y)

    test_size = int(.5 * len(data_loader.X))
    validation_size = int(.2 * .5 * len(data_loader.X))
    data_loader.split_data(validation_size, test_size)
