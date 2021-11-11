"""
Trainer object for watching and optimizing this model.

Example usage:
python train.py --auto_lr_find False --gpus 1 --prior_years 0 --prior_days 90 --crypto 'bitcoin' --window_size 14 --weight_decay 5e-5
"""
import pytorch_lightning as pl
from datamodule import CryptoDataModule
from model.btc_transformer import Transformer
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class CryptoTrainer(pl.LightningModule):
    """
    """
    def __init__(self,args,
                 n_layers,
                 filter_size,
                 dropout_rate,
                 resolution):
        super().__init__()

        self.save_hyperparameters() # turns args to self.hparams



        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.resolution = resolution

        self.model =  Transformer(n_layers=n_layers,
                 filter_size=filter_size,
                 dropout_rate=dropout_rate,
                window_size=args.window_size)


    def forward(self,x):

        return self.model(x)

    def training_step(self,batch, batch_idx):
        X, y = batch

        y_pred = self(X)

        loss = F.mse_loss(y,y_pred)

        self.log('train_loss', loss, on_epoch=True)
        # TODO , on every X steps maybe examine the plot and see what I think?
        return loss

    # validation step
    def validation_step(self,batch,batch_idx):
        X, y = batch
        y_pred = self(X)

        loss = F.mse_loss(y,y_pred)

        self.log('valid_loss', loss, on_epoch=True)

        return {'predicted': y_pred, 'truth': y, 'loss': loss}

    def validation_epoch_end(self, validation_step_outputs):
        y_pred = np.array([])
        y_true = np.array([])
        for out in validation_step_outputs:
            y_pred = np.concatenate([y_pred,out['predicted'].cpu().numpy().flatten()])
            y_true = np.concatenate([y_true,out['truth'].cpu().numpy().flatten()])

        fig = plt.figure()
        plt.plot(y_pred,label='predicted')
        plt.plot(y_true,label='truth')
        plt.ylabel("% Change")
        plt.xlabel("Validation Date")
        plt.legend()

        wandb.log({"chart": fig})

        del fig

    #test step
    def test_step(self,batch,batch_idx):
        X, y = batch
        y_pred = self(X)

        loss = F.mse_loss(y,y_pred)

        self.log('test_loss', loss, on_epoch=True)

        return loss



    def configure_optimizers(self):
        optimizer =  torch.optim.Adam(self.parameters(), lr = self.learning_rate ,weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = 10)
        return  {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}


    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--learning_rate', type=float, default=3e-4)
        parser.add_argument('--n_layers', type=int,default=12)
        parser.add_argument('--filter_size', type=int,default=12)
        parser.add_argument('--weight_decay', type=float, default=3e-4)
        parser.add_argument('--dropout_rate', type=float, default=.1)


        return parser






def main():

    pl.seed_everything(42)

    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int,
                        default=32)
    parser.add_argument('--prior_years', type=int,
                        default=5)
    parser.add_argument('--prior_days', type=int,
                        default=5)
    parser.add_argument('--window_size', type=int,
                        default=28)
    parser.add_argument('--buy_thresh', type=int,
                        default=5)
    parser.add_argument('--crypto', type=str,
                        default="bitcoin")
    parser.add_argument('--values', type=str,
                        default='usd')
    parser.add_argument('--labels_to_load', type=str,
                        default='pct_change')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = CryptoTrainer.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # training
    # ------------


    checkpoint_callback = ModelCheckpoint(
        save_last = True,
    save_top_k=1,
    verbose=True,
    monitor="valid_loss",
    mode="min"
    )

    early_stopping_callback = EarlyStopping(
                       monitor='valid_loss',
                       min_delta=0.00,
                       patience=30,
                       verbose=False,
                       mode='min'
                    )

    lr_monitor = LearningRateMonitor(logging_interval= 'step')

    run = wandb.init()#entity="automated-reporting-fdl2021",project="ESTR")
    wandb_logger = WandbLogger() #where to specify TODO


    trainer = pl.Trainer().from_argparse_args(args,logger=wandb_logger,
                       callbacks=[checkpoint_callback,lr_monitor,early_stopping_callback])

    # ------------
    # data
    # ------------
    data_module = CryptoDataModule(crypto=args.crypto,
                 prior_years=args.prior_years,
                 prior_days=args.prior_days,
                 values=args.values,
                 buy_thresh=args.buy_thresh,
                 labels_to_load=args.labels_to_load,
                 window=args.window_size,
                 batch_size=args.batch_size,
            )


    data_module.setup()
    # ------------
    # model
    # ------------

    trader = CryptoTrainer(args,
                 n_layers=args.n_layers,
                 filter_size=args.filter_size,
                 dropout_rate=args.dropout_rate,
                 resolution = data_module.approx_resolution)





    if args.auto_lr_find:
        trainer.tune(trader, data_module)

    trainer.fit(trader, data_module)


    #run.log({"training_samples" : trader.train_text_table})
    #run.log({"valid_samples" : trader.valid_text_table})
    #run.log({"test_samples" : trader.test_text_table})


    wandb.finish()


if __name__ == '__main__':
    main()
