"""
Meant to be on continuously for monitoring pricing compared to predictions

TODO - update to save predictions and this results and current results table to wandb  or another service.


Another TODO is include pct_window beyond the collect code, and make what I have here a bit more versatile. 

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
from send_email import send_email
model_path = "/home/noah/jetcoin/src/jetcoin-src/2ezjy1vf/checkpoints/epoch=74-step=4274.ckpt" #"/Users/noahkasmanoff/Desktop/F21/jetcoin/src/jetcoin-src/13kfa275/checkpoints/epoch=0-step=47.ckpt"

def predict(trader):
    """
    Returns the predicted price and percent change based on the input model path's configuration
    """


    # TODO -save all these args to model, add to trader. for loading loaders
    train_loader, valid_loader, test_loader, today_loader, approx_resolution = create_dataloaders(crypto=trader.hparams.args.crypto,
                                                                                values=trader.hparams.args.values,
                                                                                batch_size=1,
                                                                                labels_to_load=trader.hparams.args.labels_to_load.split(','),
                                                                            prior_years = trader.hparams.args.prior_years,
                                                                                prior_days = trader.hparams.args.prior_days,
                                                                                window = trader.hparams.args.window_size,
                                                                                pct_window = trader.hparams.args.pct_window,
                                                                                    buy_thresh=trader.hparams.args.buy_thresh)

    if torch.cuda.is_available():
        trader.cuda(); # if available!

    trader.eval();
    for price, price_norm, moving_avg,  date, pct_change in today_loader: # bring the moving average along to automatically convert
        if torch.cuda.is_available():
            price_norm = price_norm.cuda()# if available!

#        print(trader.forward(price).item(),'||', pct_change.item())
        y_pred = trader.forward(price_norm)#.item()


        print("Predicted % change for most recent datapoint: ", 100*y_pred.item())


    cg = CoinGeckoAPI()

    last_update = cg.get_price(ids=trader.hparams.args.crypto, vs_currencies=trader.hparams.args.values,include_last_updated_at=True)['bitcoin']

    today = last_update['last_updated_at']
    current_price = last_update[trader.hparams.args.values]

    today_df = pd.DataFrame([])

    today_df['timestamp'] = [today]
    today_df['date'] = [datetime.fromtimestamp(today)]
    today_df['predicted_pct_change'] = [y_pred.item()]
<<<<<<< HEAD
    today_df['predicted_price'] = price[-2:].mean().item()*y_pred.detach().cpu().numpy() + price[-2:].mean().item()  # take the rolling pct mean, obtain price. 
=======
    today_df['predicted_price'] = [moving_avg*y_pred.detach().cpu().numpy() + moving_avg] # this uses the mean and std.. which I no longer have.
>>>>>>> f137cfba180648fd9fb9ec89bb979022d6c56982
    today_df['current_price'] = [current_price]

    today_df['model'] = [model_path] # more!
    today_df['resolution'] = [approx_resolution]
    check_at = datetime.fromtimestamp(today+approx_resolution)
    today_df['check_at'] = [check_at]

    today_df['buy'] = today_df.apply(lambda z: 1 if z['predicted_price'] > z['current_price'] else 0,axis=1)

    if today_df['buy'].values[-1] == 1:
        print("Sending email... ")
        send_email(datetime.now(),check_at)

    if os.path.exists('../bin/predicted_changes.csv'):
        online_df = pd.read_csv('../bin/predicted_changes.csv')
        online_df = online_df.append(today_df)
        online_df.to_csv('../bin/predicted_changes.csv',index=False)

    else:
        today_df.to_csv('../bin/predicted_changes.csv',index=False)

<<<<<<< HEAD
    return today_df, y_pred, price[-2:].mean().item()
=======
    return today_df, y_pred, moving_avg
>>>>>>> f137cfba180648fd9fb9ec89bb979022d6c56982

def update_model(trader,y_pred, true_pct_change):
    """
    Take a very small gradient update step based on how close our predicted percent change was to the true.

    This should allow the model to be more flexibile to real time variability.
    """
    print("Adjusting model weights...")

    true_pct_change = torch.Tensor([true_pct_change])
    if torch.cuda.is_available():
        true_pct_change = true_pct_change.cuda(); # if available!


    optimizer = torch.optim.SGD(trader.parameters(), lr=trader.learning_rate, weight_decay=trader.weight_decay)
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(y_pred,true_pct_change)
    loss.backward()
    optimizer.step()

    return trader

def monitor():

    cg = CoinGeckoAPI()
    monitoring_df = pd.DataFrame([])

    trader = CryptoTrainer.load_from_checkpoint(model_path)

    while True:
        temp_df = pd.DataFrame([]) # for the current result, append and save to moniotoring df.

        if os.path.exists('../bin/predicted_changes.csv'):
            predicted_df = pd.read_csv('../bin/predicted_changes.csv')
        else:
            predicted_df, y_pred, rolling_mean = predict(trader)

        predicted_df['check_at'] = pd.to_datetime(predicted_df['check_at'])

        if np.datetime64(datetime.now()) > predicted_df['check_at'].values[-1]:
            # ready to log what the actual price was, and make another prediction.
            current_price = cg.get_price(ids='bitcoin', vs_currencies='usd',include_last_updated_at=True)['bitcoin']['usd']
            true_pct_change = (current_price - rolling_mean) / rolling_mean
            trader = update_model(trader, y_pred, true_pct_change)

           # trader.save_checkpoint("../bin/most_recent.ckpt")


            # this y pred persists until the next current price and % change, allowing us to get the updated weights. For that reason
            # very important these lines are after the update model one.
            predicted_price = predicted_df['predicted_price'].values[-1]
            predicted_df, y_pred, rolling_mean = predict(trader) # it predicts and updates the df

            temp_df['actual_price'] = [current_price]
            temp_df['predicted_price'] = [predicted_price]
            temp_df['timestamp'] = [datetime.now()]


            monitoring_df = monitoring_df.append(temp_df)
            monitoring_df.to_csv('../bin/results.csv',index=False)

        else:
            print("Not ready yet. Current time is ", datetime.now(), 'wait until ', predicted_df['check_at'].values[-1])
            delta = int((predicted_df['check_at'].values[-1] - np.datetime64(datetime.now())) // 1e9)
            print("sleeping for ", delta , 'seconds. ')
            time.sleep(delta)


    monitoring_df = monitoring_df.append(temp_df)

    monitoring_df.to_csv('../bin/results.csv',index=False)

    return


if __name__ == '__main__':
    monitor()
