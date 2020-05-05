import h5py
import numpy as np


class SignalGenerator(object):
    def __init__(self):
        self.pred_data = None
        self.signal = None

    def load_prediction(self):
        prediction = "evaluater/predictions.hdf5"

        with h5py.File(prediction, "r") as pred:
            # List all groups
            a_group_key = list(pred.keys())[0]

            # Get the data
            self.pred_data = np.vstack(list(pred[a_group_key]))

    def generate_signal(self):
        self.signal = np.zeros([len(self.pred_data), 2])

        self.signal[self.pred_data[:, 0] < 0, 0] = -1
        self.signal[self.pred_data[:, 1] < 0, 1] = -1
        self.signal[self.pred_data[:, 0] > 0, 0] = 1
        self.signal[self.pred_data[:, 1] > 0, 1] = 1


if __name__ == '__main__':
    pass


