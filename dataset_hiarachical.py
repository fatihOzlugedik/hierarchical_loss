import numpy as np
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import sys
from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
import h5py

# Actual dataset class
class MllDataset(Dataset):

    '''MLL mil dataset class. Can be used by pytorch DataLoader '''

    def __init__(
            self,
            path_of_dataset,
            current_fold,
            aug_im_order=True,
            split=None):
   
        self.aug_im_order = aug_im_order
        self.path_of_dataset = path_of_dataset
        path_of_fold = os.path.join(path_of_dataset,f"fold_{current_fold}",f'{split}.csv')
        data_of_fold = pd.read_csv(path_of_fold)
        self.patient = data_of_fold['patient_files'].tolist()
        self.labels = data_of_fold['diagnose'].tolist()
    
        
        
    
    def get_class_distribution(self):
        class_distribution = {}
        for label in self.labels:
            if label in class_distribution:
                class_distribution[label] += 1
            else:
                class_distribution[label] = 1
        return class_distribution    

    def __len__(self):
        return len(self.labels)
    
        

    def __getitem__(self, idx):

        pat_id = self.patient[idx]
        path= os.path.join(self.path_of_dataset, pat_id+'.h5')
        with h5py.File(path, 'r') as hf:
            bag = hf['features'][()]
            
        label = self.labels[idx]
        # shuffle features by image order in bag, if desired
        if(self.aug_im_order):
            num_rows = bag.shape[0]
            new_idx = torch.randperm(num_rows)
            bag = bag[new_idx, :]


        label_regular = torch.Tensor([label]).long()

        return bag, label_regular, pat_id