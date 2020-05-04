import os

import pandas as pd
from base.base_data_loader import BaseDataLoader


class SampleBacktestDataLoader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.code = self.config['symbol']
        self.start_time = self.config['start_time']
        self.end_time = self.config['end_time']
        self.data = None

    def load_mkt_data(self):
        dir = 'data/' + self.code + '/'
        files = [str(i) + '.csv' for i in range(self.config['start_time'].month, self.config['end_time'].month + 1)]
        for i in files:
            if self.data is not None:
                self.data = pd.concat([self.data, pd.read_csv(dir + i, usecols=range(7), parse_dates=[0])])

            else:
                self.data = pd.read_csv(dir + i, usecols=range(7), parse_dates=[0])

        loc = (self.data.loc[:, 'DATE'] <= self.config['end_time']) & \
              (self.data.loc[:, 'DATE'] >= self.config['start_time'])
        self.data = self.data.loc[loc, :]

        self.data.loc[:, 'TIME'] = self.data.iloc[:, 0] + pd.to_timedelta(self.data.iloc[:, 1])
        self.data.drop(['EX', 'DATE', 'TIME_M'], axis=1, inplace=True)
        self.data.sort_values(by=['TIME'], inplace=True)

if __name__ == '__main__':
    tmp = SampleBacktestDataLoader({'symbol': 'TSLA',
                                    'start_time': pd.datetime(2019, 1, 2),
                                    'end_time': pd.datetime(2019, 1, 3)})
    tmp.load_mkt_data()

