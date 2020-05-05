import numpy as np
import pandas as pd

from data_loader.sample_backtest_data_loader import SampleBacktestDataLoader


def main():
    sample_data = SampleBacktestDataLoader({'symbol': 'TSLA',
                                                'start_time': pd.datetime(2019, 1, 2),
                                                'end_time': pd.datetime(2019, 1, 3)})
    sample_data.load_mkt_data()



if __name__ == '__main__':
    main()