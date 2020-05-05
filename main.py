from data_loader.data_loader import DataLoader
from models.hft_model import LSTM_HFT

import h5py


def main():
    data_loader = DataLoader({"LOOKBACK": 1000, "LOOKAHEAD": 1})
    data_loader.load_data("data/TSLA0107to0108.h5")
    data_loader.organize_data()

    with h5py.File('timeseriesdataset.hdf5', 'w') as f:
        x_dset = f.create_dataset('processed_data',
                                  (data_loader.X.shape[0], data_loader.X.shape[1], data_loader.X.shape[2]),
                                  dtype='float64', data=data_loader.X)
        y_dset = f.create_dataset('processed_label',
                                  (data_loader.y.shape[0], data_loader.y.shape[1], data_loader.y.shape[2]),
                                  dtype='float64', data=data_loader.y)

    validation_size = test_size = int(.2 * len(data_loader.X))
    data_loader.split_data(validation_size, test_size)

    model = LSTM_HFT({"LOOKBACK": data_loader.lookback, 'num_col': len(data_loader.columns)})
    model.fit(data_loader.X_train, data_loader.y_train, data_loader.X_val, data_loader.y_val)

    model.evaluate(data_loader.X_test, data_loader.y_test)

    model.plot_model()

    model.predict(data_loader.X_test)


if __name__ == '__main__':
    main()
