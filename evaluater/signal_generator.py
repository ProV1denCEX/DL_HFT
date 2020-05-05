import h5py
import pandas as pd

class SignalGenerator(object):
    def __init__(self):
        pass

    def generate_signal(self):
        pass


if __name__ == '__main__':
    prediction = "evaluater/predictions.hdf5"

    with h5py.File(prediction, "r") as pred:
        # List all groups
        print("Keys: %s" % pred.keys())
        a_group_key = list(pred.keys())[0]

        # Get the data
        data = list(pred[a_group_key])

        print(1)

    # test = pd.read_hdf("data/TSLA0107to0108.h5")
    with h5py.File("data/TSLA0107to0108.h5", "r") as org:
        # List all groups
        print("Keys: %s" % org.keys())
        b_group_key = list(org.keys())[0]

        # Get the data
        test = list(org[b_group_key])

        print(1)

