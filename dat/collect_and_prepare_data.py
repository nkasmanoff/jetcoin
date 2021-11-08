from datetime import datetime
from pycoingecko import CoinGeckoAPI

import numpy as np
import pandas as pd

def collect_data(prior_years, crypto, values):
    """
    Collects data from the CoinGeck Object from the current timestamp and rolls back the designated # of years to find
    historical data, packaging it all in a pandas dataframe.

    Params:
    prior_years
        5
    crypto
        btc
    values
        usd
    """

    cg = CoinGeckoAPI()

    today = cg.get_price(ids='bitcoin', vs_currencies='usd',include_last_updated_at=True)['bitcoin']['last_updated_at']
    start = int(today - prior_years*31536000) # subtract prior years


    btc_history = cg.get_coin_market_chart_range_by_id(id=crypto,vs_currency=values,include_market_cap='true',from_timestamp= start,to_timestamp=today)
    prices = np.array(btc_history['prices'])
    timestamps = prices[:,0]
    price = prices[:,1]
    dates = [datetime.fromtimestamp(timestamp // 1000) for timestamp in timestamps]
    cg_df = pd.DataFrame()
    cg_df['timestamp'] = timestamps
    cg_df['date'] = dates
    cg_df[crypto + '_price'] = price

    return cg_df

def prepare_data(prior_years=5,crypto='bitcoin',values='usd', buy_thresh = .05, window = 7):
    """
    Converts data into a PyTorch ready format as we move feed this and variations into data loaders.

    Params:
        prior_years : int
            # of years preceeding current date to collect data for.

        crypto : str
            What cryptocurrenc(ies) to track data for. Currently only works for 1.

        buy_thresh : float
            pct_change we want to invest in prior to seeing happend.

        window : int
            # of days to compute rolling average for.

    Returns:
        coin_json : list
            list of dictionaries containing datapoints to train on.

    """

    coin_df = collect_data(prior_years=prior_years,crypto=crypto,values=values)
    coin_df['moving_avg'] = coin_df['bitcoin_price'].rolling(window=window).mean().shift(1) # for day i, computes the window rolling average for the prior i-1 to i-1-window days.
    #coin_df.dropna(inplace=True)
    coin_df['pct_change'] =  (coin_df['bitcoin_price'] - coin_df['moving_avg']) / coin_df['moving_avg']


    coin_json = []
    for i in range(window,coin_df.shape[0]):
        pct_change = coin_df['pct_change'].values[i]
        prices = coin_df['bitcoin_price'].values[i-window:i]
        date = str(coin_df['date'].values[i]).split('T')[0]

        coin_json.append({
            'pct_change':pct_change,
            'prices':prices,
            'date':date
        })

    #assign labels
    for sample in coin_json:
        if sample['pct_change'] >= buy_thresh:
            sample['y'] = 1
        else:
            sample['y'] = 0

    # create and save splits of data.
    coin_train = []
    coin_valid = []
    coin_test = []
    for sample in coin_json:
        if sample['date'].split('-')[0] == '2021':
            coin_test.append(sample)

        if sample['date'].split('-')[0] in ['2019','2020']:
            coin_valid.append(sample)
        else:
            coin_train.append(sample)

    #last, create coin today. The latest point
    coin_today = []
    prices = coin_df['bitcoin_price'].values[coin_df.shape[0]-window:coin_df.shape[0]+1]
    date = 'today'
    coin_today.append({'prices':prices,
                       'date':date,
                       'pct_change':np.nan})

    return coin_train, coin_valid, coin_test, coin_today

# TODO - test function, does this return latest date if I ask? Something like that.
