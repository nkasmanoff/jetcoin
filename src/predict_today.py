"""
Return and save a prediction for what the expected change in % is today.
"""


from train import CryptoTrainer
import torch
from dataloaders import create_dataloaders


model_path = "/Users/noahkasmanoff/Desktop/F21/jetcoin/src/uncategorized/l1h8tjje/checkpoints/epoch=28-step=986.ckpt"


trader = CryptoTrainer.load_from_checkpoint(model_path)


train_loader, valid_loader, test_loader, today_loader = create_dataloaders(crypto='bitcoin',values='usd',batch_size=1,labels_to_load=['pct_change'],
                                                            prior_years = 5, window = 28, buy_thresh=5)
trader.eval();

with torch.no_grad():
    for price, pct_change in today_loader:

#        print(trader.forward(price).item(),'||', pct_change.item())
        y_pred = trader.forward(price).item()

print("Predicted % change today: ", 100*y_pred)
# TODO use stored args for prior years etc.
print("Note caveats such as what time this is done at compared to stored price, among many other factors. ")
