from datetime import datetime
from pycoingecko import CoinGeckoAPI

import numpy as np
import pandas as pd

def collect_data(prior_years, prior_days, crypto, values):
    """
    Collects data from the CoinGecko Object from the current timestamp and rolls back the designated # of years to find
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

    today = cg.get_price(ids=crypto, vs_currencies=values,include_last_updated_at=True)[crypto]['last_updated_at']
    start = int(today - prior_years*31536000 - prior_days*86400) # subtract prior years and days


    btc_history = cg.get_coin_market_chart_range_by_id(id=crypto,vs_currency=values,include_market_cap='true',from_timestamp= start,to_timestamp=today)
    prices = np.array(btc_history['prices'])
    timestamps = prices[:,0]
    price = prices[:,1]
    #if prior_years == 0:
        #dates = [datetime.fromtimestamp(timestamp) for timestamp in timestamps]
    #else:
    # TODO - is only TODAY resolution altered when prior years 0? Either way this value is kind of irrelevant until at inference prediction.
    dates = [datetime.fromtimestamp(timestamp // 1000) for timestamp in timestamps]
    cg_df = pd.DataFrame()
    cg_df['timestamp'] = timestamps
    cg_df['date'] = dates
    cg_df[crypto + '_price'] = price


    approx_resolution = (cg_df['timestamp'].diff() / 1000).dropna().mean()

    return cg_df, approx_resolution

def prepare_data(prior_years=5, prior_days = 7,crypto='bitcoin',values='usd', buy_thresh = .05, window = 7, pct_window = 2):
    """
    Converts data into a PyTorch ready format as we move feed this and variations into data loaders.

    Params:
        prior_years : int
            # of years preceeding current date to collect data for.

        prior_days : int
            # of days precedding current date to collect data for (increases resolution if data is only in days, not years)
            Up until 90 days, can do hourly estimates.
        crypto : str
            What cryptocurrenc(ies) to track data for. Currently only works for 1.

        buy_thresh : float
            pct_change we want to invest in prior to seeing happend.

        window : int
            # of days to collect data for

        pct_window : int
            # of units to compute the percent change for, thus what we're investing with.

    Returns:
        coin_json : list
            list of dictionaries containing datapoints to train on.

    """

    coin_df,approx_resolution = collect_data(prior_years=prior_years,prior_days=prior_days,crypto=crypto,values=values)
    coin_df['moving_avg'] = coin_df[crypto+'_price'].rolling(window=pct_window).mean().shift(1) # for day i, computes the window rolling average for the prior i-1 to i-1-window units.
    #coin_df.dropna(inplace=True)
    coin_df['pct_change'] =  (coin_df[crypto+'_price'] - coin_df['moving_avg']) / coin_df['moving_avg']


    coin_json = []
    for i in range(window,coin_df.shape[0]):
        pct_change = coin_df['pct_change'].values[i]
        prices = coin_df[crypto+'_price'].values[i-window:i]
        moving_avg = coin_df['moving_avg'].values[i]
        date = str(coin_df['date'].values[i]).split('T')[0]

        coin_json.append({
            'pct_change':pct_change,
            'prices':prices,
            'moving_avg': moving_avg,
            'date':date

        })

    #assign labels
    for sample in coin_json:
        if sample['pct_change'] >= buy_thresh:
            sample['y'] = 1
        else:
            sample['y'] = 0

    # create and save splits of data.

    split1 = int(len(coin_json) * .85) # training 85 %
    split2 = int(len(coin_json)* .15) # validation 15 %
    # note I have no test set. This is fine for now, but an issue later for sure.


    coin_train = coin_json[:split1]
    coin_valid = coin_json[split1:split1+split2]
    coin_test = coin_json[split1+split2:]


    coin_today = []
    prices = coin_df[crypto+'_price'].values[coin_df.shape[0]-window:coin_df.shape[0]+1]
    date = 'today'
    moving_avg = coin_df['moving_avg'].values[-1] # last avg price
    coin_today.append({'prices':prices,
                       'date':date,
                       'moving_avg': moving_avg,
                       'pct_change':np.nan})

    return coin_train, coin_valid, coin_test, coin_today, approx_resolution

# TODO - test function, does this return latest date if I ask? Something like that.
