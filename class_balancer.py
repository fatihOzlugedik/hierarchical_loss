import numpy as np
import hiarachical_loss as hl



class Balancer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.hiararchy = {
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
        self.hierarchical_loss = hl.HierarchicalLoss(self.hiararchy, device='cuda') # or 'cpu'
        self.hierarchy_graph = self.hierarchical_loss.hierarchy_graph
        
        def get_leaf_class_distribution(self):
        """
        Get the distribution of leaf classes in the dataset.
        
        Returns:
            dict: A dictionary with leaf class names as keys and their counts as values.
        
        """
        distribution = {}
        for sample in self.dataset:
            label = sample['label']
            if label not in distribution:
                distribution[label] = 0
            distribution[label] += 1
        return distribution
    
    
    def get_mid_class_distribution(self):
        """
        Get the distribution of mid-level classes in the dataset.
        
        Returns:
            dict: A dictionary with mid-level class names as keys and their counts as values.
        
        """
        distribution = {}
        for sample in self.dataset:
            label = sample['label']
            mid_label = self.get_mid_label(label)
            if mid_label not in distribution:
                distribution[mid_label] = 0
            distribution[mid_label] += 1
        return distribution
    def get_top_class_distribution(self):
        """ 
        Get the distribution of top-level classes in the dataset.
        Returns:
            dict: A dictionary with top-level class names as keys and their counts as values.
        
        """     
        distribution = {}
        for sample in self.dataset:
            label = sample['label']
            top_label = self.get_top_label(label)
            if top_label not in distribution:
                distribution[top_label] = 0
            distribution[top_label] += 1
        return distribution
    
    def get_mid_label(self, leaf_label):
        """
        Get the mid-level label for a given leaf label.
        
        Args:
            leaf_label (str): The leaf label to find the mid-level label for.
        
        Returns:
            str: The mid-level label corresponding to the leaf label.
        """
        return list(self.hierarchy_graph.predecessors(leaf_label))[0] if leaf_label in self.hierarchy_graph else None
    
    def get_top_label(self, leaf_label):
        """
        Get the top-level label for a given leaf label.
        
        Args:
            leaf_label (str): The leaf label to find the top-level label for.
        
        Returns:
            str: The top-level label corresponding to the leaf label.
        """
        mid_label = self.get_mid_label(leaf_label)
        return list(self.hierarchy_graph.predecessors(mid_label))[0] if mid_label in self.hierarchy_graph else None
    
    def get_training_weights_leaf(self):
        """
        Get the class weights for leaf classes based on their distribution.
        
        Returns:
            dict: A dictionary with leaf class names as keys and their weights as values.
        """
        distribution = self.get_leaf_class_distribution()
        total_samples = sum(distribution.values())
        for label,count in distribution.items():
            weight
            
        
        
