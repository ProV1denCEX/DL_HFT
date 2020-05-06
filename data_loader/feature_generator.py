import pandas as pd
import numpy as np
import os


class FeatureGenerator(object):
    def __init__(self, symbol):
        self.symbol = symbol
        self.quote_dir = 'data/TAQ/' + symbol + '_quote/'
        self.trade_dir = 'data/TAQ/' + symbol + '_trade/'
        self.data = None

    def time_to_datetime(self):
        self.data['TIME'] = self.data['DATE'].astype("str").str.cat(self.data['TIME_M'])  # Combine Date&Time to one column
        self.data['TIME'] = pd.to_datetime(self.data['TIME'], format="%Y%m%d%H:%M:%S.%f")  # Change type from string to datetime
        self.data = self.data.drop(columns=['TIME_M', 'DATE']).set_index('TIME')

    def load_trade(self):
        files_to_read = sorted(os.listdir(self.trade_dir))

        for file in files_to_read:
            data = pd.read_csv(self.trade_dir + file)
            data['BID'] = data.loc[:, 'PRICE']
            data['ASK'] = data.loc[:, 'PRICE']
            data['BIDSIZ'] = data.loc[:, 'SIZE']
            data['ASKSIZ'] = data.loc[:, 'SIZE']
            data['TradeLabel'] = 1
            data = data.loc[:, ['BID', 'BIDSIZ', 'ASK', 'ASKSIZ', 'SYM_ROOT', 'TIME_M', 'DATE', 'TradeLabel']]

            if self.data is not None:
                self.data = pd.concat([self.data, data], ignore_index=True)

            else:
                self.data = data

    def load_quote(self):
        files_to_read = sorted(os.listdir(self.quote_dir))

        for file in files_to_read:
            quote = pd.read_csv(self.quote_dir + file)
            quote.loc[(quote.ASK == 0) | (quote.BID == 0), ['ASK',
                                                            'BID']] = np.nan  # Since 0 ask or bid price are outliers, turn them into nan
            quote = quote.dropna(subset=['BID', 'ASK'])  # drop rows have 0 bid or ask price
            quote = quote.loc[:, ['BID', 'BIDSIZ', 'ASK', 'ASKSIZ', 'TIME_M', 'DATE']]  # Only columns we need
            quote.loc[:, 'TradeLabel'] = 0

            if self.data is not None:
                self.data = pd.concat([self.data, quote], ignore_index=True)

            else:
                self.data = quote

    def raw_processing(self):
        self.data.loc[:, 'TimeInt'] = self.data.index.astype('int')
        temp = self.data['TimeInt'].diff()
        temp = temp.replace(0, np.nan).fillna(method='ffill').fillna(100)  # convert data has no time change with last data to 0
        self.data.loc[:, 'TimeChange'] = temp
        self.data.loc[self.data.TimeChange > 19800000000000, 'TimeChange'] = 100  # everyday is a new beginning
        self.data.drop(columns=['TimeInt'], inplace=True)
        self.data.loc[:, 'Spread '] = self.data.loc[:, 'ASK'] - self.data.loc[:, 'BID']  # new features we need in the model

    def bid_ask_balance(self):
        self.data['BidAskImbalance'] = (self.data.loc[:, 'BIDSIZ'] - self.data.loc[:, 'ASKSIZ']) / (
                    self.data.loc[:, 'BIDSIZ'] + self.data.loc[:, 'ASKSIZ'])  # new features we need in the model

    def TWAP(self):
        self.data['TwapAsk'] = (self.data['TimeChange'] * self.data['ASK']).groupby(by=[self.data.index.day]).cumsum() / self.data[
            'TimeChange'].groupby(by=[self.data.index.day]).cumsum()
        # Calculated new features, we used sum of time change times price at that certain time point divided by sum of the time change till the same time
        self.data['TwapBid'] = (self.data['TimeChange'] * self.data['BID']).groupby(by=[self.data.index.day]).cumsum() / self.data[
            'TimeChange'].groupby(by=[self.data.index.day]).cumsum()

    def VWAP(self):
        self.data['VwapAsk'] = (self.data['ASKSIZ'] * self.data['ASK']).groupby(by=[self.data.index.day]).cumsum() / self.data['ASKSIZ'].groupby(
            by=[self.data.index.day]).cumsum()
        # New features we used in model,we used sum of price*volume at that certain time point divided by sum of the volume till same time
        self.data['VwapBid'] = (self.data['BIDSIZ'] * self.data['BID']).groupby(by=[self.data.index.day]).cumsum() / self.data['BIDSIZ'].groupby(
            by=[self.data.index.day]).cumsum()

    def standardize(self):
        self.data = (self.data - self.data.mean()) / self.data.std()  # standardize the data‘

    def normalize(self):
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

    def direction(self):
        # data['BidTemp'] = np.nan
        # data['AskTemp'] = np.nan
        for i in set(self.data.index.day):
            self.data.loc[(self.data.TradeLabel == 0) & (self.data.index.day == i), 'BidTemp'] = self.data.loc[
                (self.data.TradeLabel == 0) & (self.data.index.day == i), 'BID']
            self.data.loc[(self.data.TradeLabel == 0) & (self.data.index.day == i), 'AskTemp'] = self.data.loc[
                (self.data.TradeLabel == 0) & (self.data.index.day == i), 'ASK']
            self.data.loc[self.data.index.day == i, 'AskTemp'] = self.data.loc[self.data.index.day == i, 'AskTemp'].fillna(method='ffill',
                                                                                                       limit=1)
            self.data.loc[self.data.index.day == i, 'BidTemp'] = self.data.loc[self.data.index.day == i, 'BidTemp'].fillna(method='ffill',
                                                                                                       limit=1)
        self.data['Direction'] = self.data.loc[:, 'BidTemp'] + self.data.loc[:, 'AskTemp'] - 2 * self.data.loc[:, 'BID']
        self.data.loc[:, 'Direction'] = self.data.loc[:, 'Direction'].fillna(0)
        self.data.loc[self.data.TradeLabel == 0, 'Direction'] = 0
        self.data.loc[self.data.Direction > 0, 'Direction'] = 1
        self.data.loc[self.data.Direction < 0, 'Direction'] = -1  # Close to Ask = 1, Close to Bid = -1

    @staticmethod
    def ma(data):
        miu = data.rolling(1000).mean()
        re = 0
        for i in range(999):
            miu[i] = (data[i] + re) / (i + 1)
            re += data[i]
        return miu

    def bias(self):
        # data['TPTemp'] = np.nan
        # data['BidMovingAverage'] = np.nan
        # data['AskMovingAverage'] = np.nan
        for i in set(self.data.index.day):
            self.data.loc[(self.data.TradeLabel == 1) & (self.data.index.day == i), 'TPTemp'] = self.data.loc[
                (self.data.TradeLabel == 1) & (self.data.index.day == i), 'BID']
            self.data.loc[(self.data.index.day == i), 'BidMovingAverage'] = self.ma(self.data.loc[(self.data.index.day == i), 'BID'])
            self.data.loc[(self.data.index.day == i), 'AskMovingAverage'] = self.ma(self.data.loc[(self.data.index.day == i), 'ASK'])
            self.data.loc[self.data.index.day == i, 'TPTemp'] = self.data.loc[self.data.index.day == i, 'TPTemp'].fillna(method='ffill')
        self.data.loc[:, 'TPTemp'] = self.data.loc[:, 'TPTemp'].fillna(0)
        # BidMovingAverage = data['BID'].groupby(by = [data.index.day]).apply(MovingAverage)
        # AskMovingAverage = data['ASK'].groupby(by = [data.index.day]).apply(MovingAverage)
        self.data['BidBias'] = (self.data.loc[:, 'TPTemp'] - self.data['BidMovingAverage']) / self.data['BidMovingAverage']
        self.data['AskBias'] = (self.data.loc[:, 'TPTemp'] - self.data['AskMovingAverage']) / self.data['AskMovingAverage']
        self.data.loc[self.data.TPTemp == 0, ['AskBias', 'BidBias']] = 0


if __name__ == '__main__':
    feature = FeatureGenerator("TSLA")
    feature.load_quote()
    feature.time_to_datetime()
    feature.bid_ask_balance()
    feature.TWAP()
    feature.VWAP()
    feature.normalize()
    feature.data.to_hdf('TSLA0107to0108.h5', complib='zlib', complevel=9, key='Jan')

