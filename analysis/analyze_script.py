import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hiarachical_loss as hl

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
        correct_predictions = np.trace(confusion_matrix)
        total_predictions = np.sum(confusion_matrix)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy

def main():
    # Get file path from user
    result_path = input("Enter the path to result directory: ")
    
    # Create analyzer
    analyzer = Analyzer(result_path)
    
    # Load leaf-level confusion matrices
    leaf_confusion_matrices = analyzer.load_results()
    
    # Calculate middle-level confusion matrices
    mid_confusion_matrices = analyzer.calculate_middle_confusion_matrix(leaf_confusion_matrices)
    
    # Calculate top-level confusion matrices
    top_confusion_matrices = analyzer.calculate_top_confusion_matrix(mid_confusion_matrices)
    
    # Calculate accuracies for all folds and all levels
    leaf_accuracies = []
    mid_accuracies = []
    top_accuracies = []
    
    for i in range(len(leaf_confusion_matrices)):
        # Calculate leaf-level accuracy
        leaf_acc = analyzer.calulate_accuracy(leaf_confusion_matrices[i])
        leaf_accuracies.append(leaf_acc)
        
        # Calculate middle-level accuracy
        mid_acc = analyzer.calulate_accuracy(mid_confusion_matrices[i])
        mid_accuracies.append(mid_acc)
        
        # Calculate top-level accuracy
        top_acc = analyzer.calulate_accuracy(top_confusion_matrices[i])
        top_accuracies.append(top_acc)
        
        print(f"Fold {i}: Leaf accuracy = {leaf_acc:.4f}, Middle accuracy = {mid_acc:.4f}, Top accuracy = {top_acc:.4f}")
    
    # Calculate mean and std for each level
    leaf_mean = np.mean(leaf_accuracies)
    leaf_std = np.std(leaf_accuracies)
    
    mid_mean = np.mean(mid_accuracies)
    mid_std = np.std(mid_accuracies)
    
    top_mean = np.mean(top_accuracies)
    top_std = np.std(top_accuracies)
    
    print("\nSummary:")
    print(f"Leaf level: Mean accuracy = {leaf_mean:.4f} ± {leaf_std:.4f}")
    print(f"Middle level: Mean accuracy = {mid_mean:.4f} ± {mid_std:.4f}")
    print(f"Top level: Mean accuracy = {top_mean:.4f} ± {top_std:.4f}")
    
    # Save results to a CSV file
    results = pd.DataFrame({
        'Fold': list(range(5)),
        'Leaf_Accuracy': leaf_accuracies,
        'Middle_Accuracy': mid_accuracies,
        'Top_Accuracy': top_accuracies
    })
    
    results.to_csv(f"{result_path}/hierarchical_accuracies.csv", index=False)
    print(f"Results saved to {result_path}/hierarchical_accuracies.csv")
    
    # Visualize the results
    plt.figure(figsize=(10, 6))
    x = np.arange(len(leaf_accuracies))
    width = 0.25
    
    plt.bar(x - width, leaf_accuracies, width, label='Leaf Level')
    plt.bar(x, mid_accuracies, width, label='Middle Level')
    plt.bar(x + width, top_accuracies, width, label='Top Level')
    
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Hierarchical Accuracy by Fold')
    plt.xticks(x, [f"Fold {i}" for i in range(5)])
    plt.legend()
    plt.savefig(f"{result_path}/accuracy_comparison.png")
    print(f"Visualization saved to {result_path}/accuracy_comparison.png")

if __name__ == "__main__":
    main()






