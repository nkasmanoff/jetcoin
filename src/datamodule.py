"""
Location of PyTorch-Lightning data module object used to load in the
various data loaders I plan to use.

"""
import sys
sys.path.append('../src')
from dataloaders import create_dataloaders
import pytorch_lightning as pl

class CryptoDataModule(pl.LightningDataModule):
    def __init__(self,
                 crypto: str = "bitcoin",
                 prior_years: int = 5,
                 prior_days: int = 7,
                 values: str = 'usd',
                 buy_thresh: int = 3,
                 labels_to_load: str = 'pct_change',
                 window: int = 14,
                 pct_window: int = 2,
                 batch_size: int = 32):
        super().__init__()
        self.crypto = crypto
        self.prior_years = prior_years
        self.prior_days = prior_days
        self.values = values
        self.buy_thresh = buy_thresh
        self.labels_to_load = labels_to_load.split(',')
        self.window = window
        self.pct_window = pct_window
        self.batch_size = batch_size

    def setup(self):
        self.train_loader, self.val_loader, self.test_loader, self.today_loader, self.approx_resolution = create_dataloaders(prior_years=self.prior_years,
                                                                                    prior_days = self.prior_days,
                                                                                  crypto=self.crypto,
                                                                                  values=self.values,
                                                                                  batch_size=self.batch_size,
                                                                                  buy_thresh=self.buy_thresh,
                                                                                  labels_to_load=self.labels_to_load,
                                                                                  window=self.window,
                                                                                  pct_window=self.pct_window)


    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def today(self):
        return self.today_loader
