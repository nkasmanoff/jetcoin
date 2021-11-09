import sys
import torch
from torch.utils.data import Dataset, DataLoader


sys.path.append('../dat/')

from collect_and_prepare_data import prepare_data




class CryptoDataset(Dataset):
    """
    Dataset class for crypto data.

    Reads in the list of historical weeks and depending on label to load (currently either pct_change or 'y')
    returns those as a vector to pair with input week of prices X.

    Current pre-processing step done within this class is to min-max scale the input X, and then normalize.

    Idea is gaussian input, roughly gaussian output (pct changes are kinda around that)

    Open to suggestions.


    """
    def __init__(self, coin_file,label_to_load = ['pct_change'],scale = True):
        self.coin_file = coin_file
        self.label_to_load = label_to_load

    def __len__(self):
        return len(self.coin_file)

    def __getitem__(self, idx):

        X = self.coin_file[idx]['prices']
        X_scaled = (X - X.min()) / (X.max() - X.min())
        X_norm = torch.Tensor((X_scaled - X_scaled.mean()) / X_scaled.std())


        y = []
        for label in self.label_to_load:
            y.append(self.coin_file[idx][label])


        # TODO - return date as well? Do I think I'll even care?

        return X_norm, torch.Tensor(y)

def create_dataloaders(prior_years,prior_days,crypto,values,buy_thresh,window,batch_size,labels_to_load):
    """
    Runs full data loading and preparation pipeline to allow me to experiment with all aspects of
    the datasets as part of training.



    """


    coin_train, coin_valid, coin_test, coin_today , approx_resolution = prepare_data(prior_years=prior_years,
                                                                  prior_days=prior_days,
                                                            crypto=crypto,values=values,
                                                            buy_thresh = buy_thresh,
                                                             window = window)


    train_dataset = CryptoDataset(coin_train,labels_to_load)
    valid_dataset = CryptoDataset(coin_valid,labels_to_load)
    test_dataset = CryptoDataset(coin_test,labels_to_load)
    today_dataset = CryptoDataset(coin_today,labels_to_load)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers = 4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers = 4)

    today_loader = DataLoader(today_dataset, batch_size=1, shuffle=False)



    return train_loader, valid_loader, test_loader,today_loader, approx_resolution


# TODO _ some form of pytest for CI/CD.
# example create_dataloaders(prior_years=10,crypto='btc',values='usd',buy_thresh=10,window=7,batch_size=32,labels_to_load = ['pct_change'])
