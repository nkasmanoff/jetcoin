"""
Meant to be on continuously for monitoring pricing compared to predictions
"""


from train import CryptoTrainer
import torch
from dataloaders import create_dataloaders
import pandas as pd
import os
from pycoingecko import CoinGeckoAPI
from datetime import datetime
import numpy as np
import time

model_path = "/Users/noahkasmanoff/Desktop/F21/jetcoin/src/jetcoin-src/u6ml995c/checkpoints/epoch=18-step=968.ckpt"

def predict(model_path):
    """
    Returns the predicted price and percent change based on the input model path's configuration
    """

    trader = CryptoTrainer.load_from_checkpoint(model_path)


    # TODO -save all these args to model, add to trader. for loading loaders
    train_loader, valid_loader, test_loader, today_loader, approx_resolution = create_dataloaders(crypto=trader.hparams.args.crypto,
                                                                                values=trader.hparams.args.values,
                                                                                batch_size=1,
                                                                                labels_to_load=trader.hparams.args.labels_to_load.split(','),
                                                                            prior_years = trader.hparams.args.prior_years,
                                                                                prior_days = trader.hparams.args.prior_days,
                                                                                window = trader.hparams.args.window_size,
                                                                                    buy_thresh=trader.hparams.args.buy_thresh)

    if torch.cuda.is_available():
        trader.cuda(); # if available!

    trader.eval();
    with torch.no_grad():
        for price, price_norm, date, pct_change in today_loader:
            if torch.cuda.is_available():
                price_norm = price_norm.cuda()# if available!

    #        print(trader.forward(price).item(),'||', pct_change.item())
            y_pred = trader.forward(price_norm).item()


        print("Predicted % change for most recent datapoint: ", 100*y_pred)


    cg = CoinGeckoAPI()

    today = cg.get_price(ids=trader.hparams.args.crypto, vs_currencies=trader.hparams.args.values,include_last_updated_at=True)['bitcoin']['last_updated_at']
    today_df = pd.DataFrame([])

    today_df['timestamp'] = [today]
    today_df['date'] = [datetime.fromtimestamp(today)] if trader.hparams.args.prior_years == 0 else [datetime.fromtimestamp(today // 1000)]
    today_df['predicted_pct_change'] = [y_pred]
    today_df['predicted_price'] = price.mean().item()*y_pred + price.mean().item()  # this uses the mean and std.. which I no longer have.
    today_df['model'] = [model_path] # more!
    today_df['resolution'] = [approx_resolution]
    today_df['check_at'] = [datetime.fromtimestamp(today+approx_resolution)] if trader.hparams.args.prior_years == 0 else [datetime.fromtimestamp(today // 1000)]

    if os.path.exists('../bin/predicted_changes.csv'):
        online_df = pd.read_csv('../bin/predicted_changes.csv')
        online_df = online_df.append(today_df)
        online_df.to_csv('../bin/predicted_changes.csv',index=False)

    else:
        today_df.to_csv('../bin/predicted_changes.csv',index=False)

    return today_df




i = 0
monitoring_df = pd.DataFrame([])
while i < 10:
    predicted_prices = []
    actual_prices = []
    timestamps = []

    if os.path.exists('../bin/predicted_changes.csv'):
        predicted_df = pd.read_csv('../bin/predicted_changes.csv')
    else:
        predicted_df = predict(model_path)


    predicted_df['check_at'] = pd.to_datetime(predicted_df['check_at'])

    if np.datetime64(datetime.now()) > predicted_df['check_at'].values[-1]:
        # ready to log what the actual price was, and make another prediction.
        current_price = cg.get_price(ids='bitcoin', vs_currencies='usd',include_last_updated_at=True)['bitcoin']['usd']
        predicted_price = predicted_df['predicted_price'].values[-1]
        predicted_df = predict(model_path)

        actual_prices.append(current_price)
        predicted_prices.append(predicted_price)
        timestamps.append(datetime.now())

        i += 1

    else:
        print("Not ready yet. Current time is ", datetime.now(), 'wait until ', predicted_df['check_at'].values[-1])
        delta = int((predicted_df['check_at'].values[-1] - np.datetime64(datetime.now())) // 1e9)
        print("sleeping for ", delta , 'seconds. ')
        time.sleep(delta)


monitoring_df['predicted_prices'] = predicted_prices
monitoring_df['actual_prices'] = actual_prices
monitoring_df['timestamps'] = timestamps

monitoring_df.to_csv('../bin/results.csv',index=False)
