import torch.utils.data as data
import torch
import random
import h5py
import json

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')
        self.order = []
        self.indices = range(0, self.data.shape[0])

        with open("hdf5.json", "r") as config:
            self.total_patches = json.load(config)["total_patches"]

    def __getitem__(self, index):
        if not self.order:
            self.order = random.sample(self.indices, self.total_patches)
        hdf5_index = self.order.pop(0)
        return (torch.from_numpy(self.data[hdf5_index,:,:,:]).float(),
                torch.from_numpy(self.target[hdf5_index,:,:,:]).float())
        
    def __len__(self):
        return self.total_patches
