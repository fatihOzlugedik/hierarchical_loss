from model_train_hiarachical import *   # model training function
from dataset_hiarachical import *       # dataset
# makes conversion from string label to one-hot encoding easier
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.multiprocessing
import torch
import sys
import os
import time
import argparse as ap
import models
from models.transformer import Transformer
from models.wbcmil import WBCMIL

torch.multiprocessing.set_sharing_strategy('file_system')
class_count = 8
# import from other, own modules
# get the number of patients in each class counts
def get_class_sizes(dictionary):
    class_sizes = []
    class_labels = [0,1,2,3,4,5,6,7]
    
    for class_label in class_labels:
        count = dictionary.get(class_label,0)  # Avoid KeyError
        class_sizes.append(count)
    
    return class_sizes





# get arguments from parser, set up folder
# parse arguments
parser = ap.ArgumentParser()

# Algorithm / training parameters
parser.add_argument(
    '--fold',
    help='offset for cross-validation (1-5). Change to cross-validate',
    required=True,
    )  

parser.add_argument(
    '--arch',
    required=True,
    default='wbcmil',
    choices = ['transformer', 'wbcmil'])  # shift folds for cross validation. Increasing by 1 moves all folds by 1.

parser.add_argument(
    '--lr',
    help='used learning rate',
    required=False,
    default=0.00005)                                     # learning rate
parser.add_argument(
    '--ep',
    help='max. amount after which training should stop',
    required=False,
    default=150)               # epochs to train
parser.add_argument(
    '--es',
    help='early stopping if no decrease in loss for x epochs',
    required=False,
    default=20)          # epochs without improvement, after which training should stop.
parser.add_argument(
    '--multi_att',
    help='use multi-attention approach',
    required=False,
    default=1)                          # use multiple attention values if 1

# Data parameters: Modify the dataset
parser.add_argument(
    '--prefix',
    help='define which set of features shall be used',
    required=False,
    default='fnl34_')        # define feature source to use (from different CNNs)
# pass -1, if no filtering acc to peripheral blood differential count
# should be done
parser.add_argument(
    '--filter_diff',
    help='Filters AML patients with less than this perc. of MYB.',
    default=20) #previously set to 20 
# Leave out some more samples, if we have enough without them. Quality of
# these is not good, but if data is short, still ok.
parser.add_argument(
    '--filter_mediocre_quality',
    help='Filters patients with sub-standard sample quality',
    default=0)
parser.add_argument(
    '--bootstrap_idx',
    help='Remove one specific patient at pos X',
    default=-
    1)                             
parser.add_argument(
    '--result_folder',
    help='store folder with custom name',
    required=False)                                 
parser.add_argument(
    '--save_model',
    help='choose wether model should be saved',
    required=False,
    default=1)           
               # store model parameters if 1
args = parser.parse_args()
fold=args.fold

dataset_path="/vol/data/Beluga"
TARGET_FOLDER=f"/vol/data/Belgua_results/baseline_superBloom_{args.arch}"
# store results in target folder
TARGET_FOLDER = os.path.join(TARGET_FOLDER, f'fold_{fold}')
if not os.path.exists(TARGET_FOLDER):
    os.makedirs(TARGET_FOLDER)
start = time.time()

datasets = {}

datasets['train'] = MllDataset(
  
    path_of_dataset=dataset_path,
    current_fold=int(fold),
    aug_im_order=True,
    split='train',
)


datasets['val'] = MllDataset(
    path_of_dataset=dataset_path,
    current_fold=int(fold),
    aug_im_order=True,
    split='val',
)


datasets['test'] = MllDataset(
    path_of_dataset=dataset_path,
    current_fold=int(fold),
    aug_im_order=False,
    split='test'
)




# Initialize dataloaders
print("Initialize dataloaders...")
dataloaders = {}

# Ensure balanced sampling
# Get total sample sizes
#class_sizes_train = get_class_sizes(datasets['train'].get_class_distribution())
#class_sizes_val = get_class_sizes(datasets['val'].get_class_distribution())
#class_sizes_test = get_class_sizes(datasets['test'].get_class_distribution())

# Sum the class sizes from the training, validation, and test splits
#class_sizes_total = [train + val + test for train, val, test in zip(class_sizes_train, class_sizes_val, class_sizes_test)]

print("Total class sizes:", class_sizes_total)

# Calculate label frequencies
#label_freq = [class_sizes_total[c] / sum(class_sizes_total) for c in range(class_count)]
# Balance sampling frequencies for equal sampling
#individual_sampling_prob = [(1 / class_count) * (1 / label_freq[c]) for c in range(class_count)]

print(datasets['train'])

#idx_sampling_freq_train = torch.tensor(individual_sampling_prob)[datasets['train'].labels]

#sampler_train = WeightedRandomSampler(
    #weights=idx_sampling_freq_train,
    #replacement=True,
    #num_samples=len(idx_sampling_freq_train)
#)

dataloaders['train'] = DataLoader(
    datasets['train'],
    #sampler=sampler_train,    
    num_workers=4  
)

dataloaders['val'] = DataLoader(
datasets['val'],
num_workers=4  
)

dataloaders['test'] = DataLoader(datasets['test'])
print("")


# 3: Model
# initialize model, GPU link, training

# set up GPU link and model (check for multi GPU setup)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ngpu = torch.cuda.device_count()
print("Found device: ", ngpu, "x ", device)


if args.arch.lower() == 'transformer':
    model = Transformer(input_dim=384, num_classes=class_count, linear_dropout=0.0)
elif args.arch.lower() == 'wbcmil':
    model = WBCMIL(input_dim=384, num_classes=class_count, linear_dropout=0.0)


if(ngpu > 1):
    model = torch.nn.DataParallel(model)
model = model.to(device)
print("Setup complete.")
print("")

optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.01)
scheduler = None

# launch training
train_obj = ModelTrainer(
    model=model,
    dataloaders=dataloaders,
    epochs=int(
        args.ep),
    optimizer=optimizer,
    scheduler=scheduler,
    class_count=class_count,
    early_stop=int(
        args.es),
    device=device)
model, conf_matrix, data_obj = train_obj.launch_training()


# 4: aftermath
# save confusion matrix from test set, all the data , model, print parameters

np.save(os.path.join(TARGET_FOLDER, 'test_conf_matrix.npy'), conf_matrix)
pickle.dump(
    data_obj,
    open(
        os.path.join(
            TARGET_FOLDER,
            'testing_data.pkl'),
        "wb"))

if(int(args.save_model)):
    torch.save(model, os.path.join(TARGET_FOLDER, 'model.pt'))
    torch.save(model, os.path.join(TARGET_FOLDER, 'state_dictmodel.pt'))

end = time.time()
runtime = end - start
time_str = str(int(runtime // 3600)) + "h" + str(int((runtime %
                                                      3600) // 60)) + "min" + str(int(runtime % 60)) + "s"

# other parameters
print("")
print("------------------------Final report--------------------------")
print('prefix', args.prefix)
print('Runtime', time_str)
print('max. Epochs', args.ep)
print('Learning rate', args.lr)
