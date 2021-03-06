import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../dat/')

from collect_and_prepare_data import prepare_data

from utils import get_bin_num, get_bins


class CryptoDataset(Dataset):
    """
    Dataset class for crypto data.

    Reads in the list of historical weeks and depending on label to load (currently either pct_change or 'y')
    returns those as a vector to pair with input week of prices X.

    Current pre-processing step done within this class is to min-max scale the input X, and then normalize.

    Idea is gaussian input, roughly gaussian output (pct changes are kinda around that)

    Open to suggestions.


    """
    def __init__(self, coin_file,label_to_load = ['pct_change'],today = False):
        self.coin_file = coin_file
        self.label_to_load = label_to_load
        self.today = today

    def __len__(self):
        return len(self.coin_file)

    def __getitem__(self, idx):

        X = self.coin_file[idx]['prices']
        X_scaled = (X - X.min()) / (X.max() - X.min())
        X_norm = torch.Tensor((X_scaled - X_scaled.mean()) / X_scaled.std())


        y = []
        for label in self.label_to_load:
            y.append(self.coin_file[idx][label])


        if self.today: #when using today loader, it returns all relevant info.
            return X, X_norm, self.coin_file[idx]['moving_avg'], self.coin_file[idx]['date'],torch.Tensor(y)

        return X_norm, torch.Tensor(y)

def create_dataloaders(prior_years,prior_days,crypto,values,buy_thresh,window,batch_size,labels_to_load, pct_window, weighted_sampling = False):
    """
    Runs full data loading and preparation pipeline to allow me to experiment with all aspects of
    the datasets as part of training.



    """


    coin_train, coin_valid, coin_test, coin_today , approx_resolution = prepare_data(prior_years=prior_years,
                                                                  prior_days=prior_days,
                                                            crypto=crypto,values=values,
                                                            buy_thresh = buy_thresh,
                                                             window = window,
                                                             pct_window = pct_window)


    train_dataset = CryptoDataset(coin_train,labels_to_load)
    valid_dataset = CryptoDataset(coin_valid,labels_to_load)
    test_dataset = CryptoDataset(coin_test,labels_to_load)
    today_dataset = CryptoDataset(coin_today,labels_to_load,today=True)


    params = [i['pct_change'] for i in coin_train]
    num_bins = 3 # fixed for now. More bins = smaller space between -> more weight to outliers as they do not get binned with others. Have noticed that this exagerates the impact of high changes; should want to be more conservative, so reducing this to 3. 
    bin_sample_counts, bin_edges = get_bins(params, num_bins)
    bin_weights = 1./torch.Tensor(bin_sample_counts)


    train_targets = [get_bin_num(bin_edges,sample) for sample in params]
    train_samples_weight = [bin_weights[bin_id] for bin_id in train_targets]
    train_samples_weight = np.array(train_samples_weight)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_samples_weight, train_dataset.__len__())
    if weighted_sampling:
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,sampler=train_sampler,num_workers = 4) # sampler is mutually exclusive with shuffle
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers = 4)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers = 4)

    today_loader = DataLoader(today_dataset, batch_size=1, shuffle=False)



    return train_loader, valid_loader, test_loader,today_loader, approx_resolution


# TODO _ some form of pytest for CI/CD.
# example create_dataloaders(prior_years=10,crypto='btc',values='usd',buy_thresh=10,window=7,batch_size=32,labels_to_load = ['pct_change'])
