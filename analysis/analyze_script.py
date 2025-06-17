import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hiarachical_loss as hl
import dataframe_image as dfi
import argparse as ap
from sklearn.metrics import f1_score, balanced_accuracy_score
import analyzer as an

def main():
    parser = ap.ArgumentParser()

    # Algorithm / training parameters
   
    parser.add_argument(
        '--alpha',
        help='alpha value for hierarchical loss',
        required=True,
    )
    parser.add_argument(
        '--arch',
        help='model architecture to use',
        required=True,
    )
    args = parser.parse_args()
    alpha = float(args.alpha)
    architecture = args.arch

    result_path = f"/lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/results_dinoBloomB/hl_{alpha}_{architecture}"
    #result_path='/lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/results_dinoBloomS/CE_wbcmil'
    # Create analyzer
    analyzer = an.Analyzer(result_path)
    
    # Load leaf-level confusion matrices
    leaf_confusion_matrices = analyzer.load_results()
    
    # Calculate middle-level confusion matrices
    mid_confusion_matrices = analyzer.calculate_middle_confusion_matrix(leaf_confusion_matrices)
    
    # Calculate top-level confusion matrices
    top_confusion_matrices = analyzer.calculate_top_confusion_matrix(mid_confusion_matrices)
    
    # Calculate metrics for all folds and all levels
    leaf_accuracies, mid_accuracies, top_accuracies = [], [], []
    leaf_bal_accs, mid_bal_accs, top_bal_accs = [], [], []
    leaf_macro_f1s, mid_macro_f1s, top_macro_f1s = [], [], []
    leaf_weighted_f1s, mid_weighted_f1s, top_weighted_f1s = [], [], []

    for i in range(len(leaf_confusion_matrices)):
        # Leaf level
        leaf_acc = analyzer.calulate_accuracy(leaf_confusion_matrices[i])
        leaf_bal = analyzer.calculate_balanced_accuracy(leaf_confusion_matrices[i])
        leaf_macro_f1 = analyzer.calculate_macro_f1(leaf_confusion_matrices[i])
        leaf_weighted_f1 = analyzer.calculate_weighted_f1(leaf_confusion_matrices[i])
        leaf_accuracies.append(leaf_acc)
        leaf_bal_accs.append(leaf_bal)
        leaf_macro_f1s.append(leaf_macro_f1)
        leaf_weighted_f1s.append(leaf_weighted_f1)

        # Middle level
        mid_acc = analyzer.calulate_accuracy(mid_confusion_matrices[i])
        mid_bal = analyzer.calculate_balanced_accuracy(mid_confusion_matrices[i])
        mid_macro_f1 = analyzer.calculate_macro_f1(mid_confusion_matrices[i])
        mid_weighted_f1 = analyzer.calculate_weighted_f1(mid_confusion_matrices[i])
        mid_accuracies.append(mid_acc)
        mid_bal_accs.append(mid_bal)
        mid_macro_f1s.append(mid_macro_f1)
        mid_weighted_f1s.append(mid_weighted_f1)

        # Top level
        top_acc = analyzer.calulate_accuracy(top_confusion_matrices[i])
        top_bal = analyzer.calculate_balanced_accuracy(top_confusion_matrices[i])
        top_macro_f1 = analyzer.calculate_macro_f1(top_confusion_matrices[i])
        top_weighted_f1 = analyzer.calculate_weighted_f1(top_confusion_matrices[i])
        top_accuracies.append(top_acc)
        top_bal_accs.append(top_bal)
        top_macro_f1s.append(top_macro_f1)
        top_weighted_f1s.append(top_weighted_f1)

        print(
            f"Fold {i}: "
            f"Leaf acc={leaf_acc:.4f}, bal_acc={leaf_bal:.4f}, macroF1={leaf_macro_f1:.4f}, wF1={leaf_weighted_f1:.4f} | "
            f"Mid acc={mid_acc:.4f}, bal_acc={mid_bal:.4f}, macroF1={mid_macro_f1:.4f}, wF1={mid_weighted_f1:.4f} | "
            f"Top acc={top_acc:.4f}, bal_acc={top_bal:.4f}, macroF1={top_macro_f1:.4f}, wF1={top_weighted_f1:.4f}"
        )

    # Calculate mean and std for each metric and level
    def mean_std(arr): return (np.mean(arr), np.std(arr))

    metrics = {
        "Leaf_Accuracy": leaf_accuracies,
        "Leaf_Balanced_Accuracy": leaf_bal_accs,
        "Leaf_MacroF1": leaf_macro_f1s,
        "Leaf_WeightedF1": leaf_weighted_f1s,
        "Middle_Accuracy": mid_accuracies,
        "Middle_Balanced_Accuracy": mid_bal_accs,
        "Middle_MacroF1": mid_macro_f1s,
        "Middle_WeightedF1": mid_weighted_f1s,
        "Top_Accuracy": top_accuracies,
        "Top_Balanced_Accuracy": top_bal_accs,
        "Top_MacroF1": top_macro_f1s,
        "Top_WeightedF1": top_weighted_f1s,
    }

    # Print summary
    print("\nSummary:")
    for key in ["Leaf", "Middle", "Top"]:
        print(
            f"{key} level: "
            f"Mean acc={np.mean(metrics[f'{key}_Accuracy']):.4f} ± {np.std(metrics[f'{key}_Accuracy']):.4f}, "
            f"Mean bal_acc={np.mean(metrics[f'{key}_Balanced_Accuracy']):.4f} ± {np.std(metrics[f'{key}_Balanced_Accuracy']):.4f}, "
            f"Mean macroF1={np.mean(metrics[f'{key}_MacroF1']):.4f} ± {np.std(metrics[f'{key}_MacroF1']):.4f}, "
            f"Mean wF1={np.mean(metrics[f'{key}_WeightedF1']):.4f} ± {np.std(metrics[f'{key}_WeightedF1']):.4f}"
        )

    # Save results to a CSV file
    results = pd.DataFrame({
        'Fold': list(range(5)),
        'Leaf_Accuracy': leaf_accuracies,
        'Leaf_Balanced_Accuracy': leaf_bal_accs,
        'Leaf_MacroF1': leaf_macro_f1s,
        'Leaf_WeightedF1': leaf_weighted_f1s,
        'Middle_Accuracy': mid_accuracies,
        'Middle_Balanced_Accuracy': mid_bal_accs,
        'Middle_MacroF1': mid_macro_f1s,
        'Middle_WeightedF1': mid_weighted_f1s,
        'Top_Accuracy': top_accuracies,
        'Top_Balanced_Accuracy': top_bal_accs,
        'Top_MacroF1': top_macro_f1s,
        'Top_WeightedF1': top_weighted_f1s,
    })

    results.to_csv(f"{result_path}/hierarchical_metrics.csv", index=False)
    print(f"Results saved to {result_path}/hierarchical_metrics.csv")

    # --- Save table as PNG with mean and std row using dataframe_image ---
    df = pd.read_csv(f"{result_path}/hierarchical_metrics.csv")
    mean_row = ['Mean'] + [df[col].mean() for col in df.columns if col != 'Fold']
    std_row = ['Std'] + [df[col].std() for col in df.columns if col != 'Fold']
    df_table = df.copy()
    df_table.loc['Mean'] = mean_row
    df_table.loc['Std'] = std_row

    # Format all float columns to 3 decimals
    for col in df_table.columns:
        if col != 'Fold':
            df_table[col] = df_table[col].apply(lambda x: f"{x:.3f}")

    dfi.export(df_table, f"{result_path}/hierarchical_metrics_table.png")
    print(f"Table saved to {result_path}/hierarchical_metrics_table.png")
    



if __name__ == "__main__":
    main()






