import numpy as np
import pandas as pd

from data_loader.sample_backtest_data_loader import SampleBacktestDataLoader
from backtester.dummy_signal_generator import DummySignalGenerator


def main():
    sample_data = SampleBacktestDataLoader({'symbol': 'TSLA',
                                                'start_time': pd.datetime(2019, 1, 2),
                                                'end_time': pd.datetime(2019, 1, 3)})
    sample_data.load_mkt_data()

    dummy_signal = DummySignalGenerator()
    dummy_signal.generate_signal(sample_data.data.loc[:, 'TIME'])


if __name__ == '__main__':
    main()