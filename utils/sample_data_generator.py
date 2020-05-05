import pandas as pd


def generate_sample_data(taq_dir, symbol, month):
    quote_data = pd.read_csv(taq_dir + symbol + "_quote/" + str(month) + ".csv")
    sample_data = quote_data.iloc[:5000, :]

    sample_data.to_csv(taq_dir + symbol + "_quote/" + str(month) + "_sample.csv", index=False)

    trade_data = pd.read_csv(taq_dir + symbol + "_trade/" + symbol + "_trade.csv")

    start_date = sample_data.iloc[0, 0]
    end_time = max(sample_data.iloc[:, 1])

    loc = (trade_data.iloc[:, 1] <= end_time) & (trade_data.iloc[:, 0] == start_date)

    sample_trade_data = trade_data.loc[loc, :]
    sample_trade_data.to_csv(taq_dir + symbol + "_trade/" + symbol + "_trade_sample.csv", index=False)


if __name__ == '__main__':
    for i in ["AAPL", "TSLA"]:
        taq_dir = "data/TAQ/"
        month = 1
        generate_sample_data(taq_dir, i, month)
