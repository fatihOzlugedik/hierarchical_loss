import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hiarachical_loss as hl
import dataframe_image as dfi
import argparse as ap
from sklearn.metrics import f1_score, balanced_accuracy_score


class Analyzer:
    
    def __init__(self,result_path):
        self.result_path = result_path
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
        hiarachical_loss = hl.HierarchicalLoss(self.hiararchy, device='cpu')
        self.T32 = hiarachical_loss.T32
        self.T21 = hiarachical_loss.T21
 
    def load_results(self):
        confusion_matrixes=[]
        for i in range(5):
            conf_matrix = np.load(f"{self.result_path}/fold_{i}/test_conf_matrix.npy")
            confusion_matrixes.append(conf_matrix)
        
        return confusion_matrixes
    
    
    def calculate_middle_confusion_matrix(self, leaf_confusion_matrixes):
        mid_conf_matrices = []
        for conf_matrix in leaf_confusion_matrixes:
            mid_conf_matrix = self.T32.T @ conf_matrix @ self.T32 
            mid_conf_matrices.append(mid_conf_matrix)
        return mid_conf_matrices


    def calculate_top_confusion_matrix(self, mid_confusion_matrixes):
        top_conf_matrices = []
        for conf_matrix in mid_confusion_matrixes:
            top_conf_matrix = self.T21.T @ conf_matrix @ self.T21 
            top_conf_matrices.append(top_conf_matrix)
        return top_conf_matrices
    
    def calulate_accuracy(self, confusion_matrix):
        # Convert to numpy if it's a torch tensor
        if hasattr(confusion_matrix, "cpu") and hasattr(confusion_matrix, "numpy"):
            confusion_matrix = confusion_matrix.cpu().numpy()
        correct_predictions = np.trace(confusion_matrix)
        total_predictions = np.sum(confusion_matrix)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy
    
    def calculate_balanced_accuracy(self, confusion_matrix):
        # Convert to numpy if it's a torch tensor
        if hasattr(confusion_matrix, "cpu") and hasattr(confusion_matrix, "numpy"):
            confusion_matrix = confusion_matrix.cpu().numpy()
        # True labels and predictions from confusion matrix
        y_true = []
        y_pred = []
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                y_true += [i] * int(confusion_matrix[i, j])
                y_pred += [j] * int(confusion_matrix[i, j])
        if len(y_true) == 0:
            return 0
        return balanced_accuracy_score(y_true, y_pred)

    def calculate_macro_f1(self, confusion_matrix):
        if hasattr(confusion_matrix, "cpu") and hasattr(confusion_matrix, "numpy"):
            confusion_matrix = confusion_matrix.cpu().numpy()
        y_true = []
        y_pred = []
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                y_true += [i] * int(confusion_matrix[i, j])
                y_pred += [j] * int(confusion_matrix[i, j])
        if len(y_true) == 0:
            return 0
        return f1_score(y_true, y_pred, average='macro', zero_division=0)

    def calculate_weighted_f1(self, confusion_matrix):
        if hasattr(confusion_matrix, "cpu") and hasattr(confusion_matrix, "numpy"):
            confusion_matrix = confusion_matrix.cpu().numpy()
        y_true = []
        y_pred = []
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                y_true += [i] * int(confusion_matrix[i, j])
                y_pred += [j] * int(confusion_matrix[i, j])
        if len(y_true) == 0:
            return 0
        return f1_score(y_true, y_pred, average='weighted', zero_division=0)