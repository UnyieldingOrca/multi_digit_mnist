'''
Created on Jun 27, 2018

@author: david
'''
from torch.utils.data.dataset import Dataset
import gzip
import pickle
import torch.utils.data


def identity(item):
    return item


def build_DataLoader(dataset, batch_size=100, shuffle=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def load_dataset(file, data_aug=identity):
    print('loading: ' + file)
    fp = gzip.open(file, 'rb')
    data = pickle.load(fp)
    return dataset(data, data_aug)


class dataset(Dataset):

    def __init__(self, data, data_aug=identity):
        assert all(data[0].shape[0] == array.shape[0] for array in data)
        self.data_aug = data_aug
        self.data = data

    def __getitem__(self, index):
        return self.data_aug(tuple(array[index] for array in self.data))

    def __len__(self):
        return self.data[0].shape[0]
    
    def save(self, file):
        print('saving: ' + file)
        fp = gzip.open(file, 'wb')
        pickle.dump(self.data, fp)
