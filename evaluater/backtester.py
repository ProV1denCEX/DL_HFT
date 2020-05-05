import h5py
import numpy as np
from evaluater.signal_generator import SignalGenerator


class Backtester(object):
    def __init__(self):
        self.data = None
        self.nv = None

    def run(self, signal):
        ret = np.diff(self.data, axis=0) / self.data[:-1, :]
        ret = ret * signal
        self.ret = ret
        ret = ret.sum(axis=1) + 1

        nv = ret.cumprod()
        self.nv = nv

    def evaluate(self):
        win_rate = sum(np.diff(self.nv) > 0) / len(self.nv)
        ret = self.nv[-1] - 1
        sharpe = ret / self.ret.std()

        print("win_rate:", win_rate)
        print("return in backtest:", ret)
        print("sharpe:", sharpe)

    def load_original_data(self):
        with h5py.File("data/TSLA0107to0108.h5", "r") as org:
            # List all groups
            a_group_key = list(org.keys())[0]
            raw_data = list(org[a_group_key].items())
            bytes_columns = raw_data[0][1][()]
            columns = [i.decode('utf-8') for i in bytes_columns]
            data_1 = raw_data[-1][1][()]

            self.data = data_1[151000:200000, [0, 2]]

            self.data[:, 0] *= 7.305569721
            self.data[:, 0] += 329.8444727

            self.data[:, 1] *= 5.271191298
            self.data[:, 1] += 334.5558983


if __name__ == '__main__':
    bt = Backtester()
    bt.load_original_data()

    signal = SignalGenerator()
    signal.load_prediction()
    signal.generate_signal()

    bt.run(signal.signal)

    bt.evaluate()

