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
import hiarachical_loss as hl

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
        self.hiararchy ={
        "root": ["Malignant", "NonMalignant"],

        "Malignant": [
        "Acute Leukemias",
        "Myelodysplastic Syndromes",
        "Myeloid Overlap Syndromes",
        "Chronic Myeloid Neoplasms",
        "Lymphoid Neoplasms",
        "Plasma Cell Neoplasms"
        ],
        "Acute Leukemias": ["AML", "ALL", "AL"],
        "Myelodysplastic Syndromes": ["MDS"],
        "Myeloid Overlap Syndromes": ["CMML", "MDS / MPN", "MPN / MDS-RS-T"],
        "Chronic Myeloid Neoplasms": ["MPN", "CML", "ET", "PV"],
        "Lymphoid Neoplasms": ["B-cell neoplasm", "HCL", "T-cell neoplasm"],
        "Plasma Cell Neoplasms": ["MM", "PCL"],

        "NonMalignant": [
        "Reactive Conditions",
        "Normal Findings"
        ],
        "Reactive Conditions": ["Reactive changes"],
        "Normal Findings": ["Normalbefund"]
        }
        self.hierarchical_loss = hl.HierarchicalLoss(self.hiararchy, device='cuda') 
        self.leaf_to_idx = self.hierarchical_loss.leaf_to_idx
        
        self.patient = data_of_fold['patient_files'].tolist()
        self.string_labels = data_of_fold['diagnose'].tolist()
        self.labels = []
        for label in string_labels:
            if label in self.leaf_to_idx:
                self.labels.append(self.leaf_to_idx[label])
            else:
                error_message = f"Label '{label}' not found in leaf_to_idx mapping."
                raise ValueError(error_message)
    
        
        
    
    def get_class_distribution(self):
       
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