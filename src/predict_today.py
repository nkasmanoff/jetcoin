"""
Return and save a prediction for what the expected change in % is today.
"""


from train import CryptoTrainer
import torch
from dataloaders import create_dataloaders
import pandas as pd
import os
from pycoingecko import CoinGeckoAPI
from datetime import datetime


model_path = "/Users/noahkasmanoff/Desktop/F21/jetcoin/src/jetcoin-src/u6ml995c/checkpoints/epoch=18-step=968.ckpt"



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
