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
        #Construct the path to the fold data
        path_of_fold = os.path.join(path_of_dataset,f"data_fold_{current_fold}",f'{split}.csv')
        data_of_fold = pd.read_csv(path_of_fold)
        
        self.patient = data_of_fold['patient_files'].tolist()
        self.labels = data_of_fold['labels'].tolist()
    
        
        
    def __len__(self):
        return len(self.labels)
    
        

    def __getitem__(self, idx):

        pat_id = self.patient[idx]
        path= os.path.join(self.path_of_dataset, pat_id+'_combined.pt')
        data = torch.load(path, map_location=torch.device("cpu"),weights_only=False)
        bag = data['features']    
        label = self.labels[idx]
        # shuffle features by image order in bag, if desired
        if(self.aug_im_order):
            num_rows = bag.shape[0]
            new_idx = torch.randperm(num_rows)
            bag = bag[new_idx, :]


        #label_regular = torch.Tensor([label]).long()

        return bag, label, pat_id