import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import hiarachical_loss as hl


import sys
import time
import copy
import numpy as np
import gc



class ModelTrainer:
    '''class containing all the info about the training process and handling the actual
    training function'''

    def __init__(
            self,
            model,
            dataloaders,
            epochs,
            optimizer,
            scheduler,
            class_count,
            device,
            loss,
            alpha,
            early_stop=20):
        self.model = model
        self.dataloaders = dataloaders
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.class_count = class_count
        self.early_stop = early_stop
        self.device = device
        self.loss = loss
        self.alpha = alpha
        self.hiararchy= {
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
        self.hiarachical_loss = hl.HierarchicalLoss(self.hiararchy, device=self.device, alpha=self.alpha)

    def launch_training(self):
        '''initializes training process.'''

        best_loss = 10  # high value, so that future loss values will always be lower
        no_improvement_for = 0
        best_model = copy.deepcopy(self.model.state_dict())

        for ep in range(self.epochs):
            # perform train/val iteration
            loss, acc, conf_matrix, data_obj = self.dataset_to_model(
                ep, 'train')
          

            loss, acc, conf_matrix, data_obj = self.dataset_to_model(ep, 'val')
            

            no_improvement_for += 1

            loss = loss.cpu().numpy()

            # if improvement, reset counter
            if(loss < best_loss):
                best_model = copy.deepcopy(self.model.state_dict())
                best_loss = loss
                no_improvement_for = 0
                print("Best!")

            # break if X times no improvement
            if(no_improvement_for == self.early_stop):
                break

            # scheduler (optional)
            if not (self.scheduler is None):
                if isinstance(
                        self.scheduler,
                        optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()

        # load best performing model, and launch on test set
        self.model.load_state_dict(best_model)
        loss, acc, conf_matrix, data_obj = self.dataset_to_model(ep, 'test')
        return self.model, conf_matrix, data_obj

    def dataset_to_model(self, epoch, split, backprop_every=20):
        '''launch iteration for 1 epoch on specific dataset object, with backprop being optional
        - epoch: epoch count, is only printed and thus not super important
        - split: if equal to 'train', apply backpropagation. Otherwise, don`t.
        - backprop_every: only apply backpropagation every n patients. Allows for gradient accumulation over
          multiple patients, like in batch processing in a regular neural network.'''

        if(split == 'train'):
            backpropagation = True
            self.model.train()
        else:
            backpropagation = False
            self.model.eval()

        # initialize data structures to store results
        corrects = 0
        train_loss = 0
        time_pre_epoch = time.time()
        confusion_matrix = np.zeros(
            (self.class_count, self.class_count), np.int16)
        data_obj = DataMatrix()

        self.optimizer.zero_grad()
        backprop_counter = 0

        for _, (bag, label, path_full) in enumerate(
                self.dataloaders[split]):


            bag = bag.to(self.device)
               
            # forward pass
            prediction= self.model(
                bag)

            
            # Get the list of leaf nodes and their indices
            leaf_nodes = list(self.hiarachical_loss.leaf_to_idx.keys())
            num_leaves = len(leaf_nodes)
    
            one_hot_label = torch.zeros(num_leaves)
            one_hot_label[label] = 1.0
           
            if self.loss == "hl":
                loss_out=self.hiarachical_loss.get_loss(prediction, one_hot_label.to(self.device))
            elif self.loss == "CE":
                loss_function = nn.CrossEntropyLoss()
                loss_out = loss_function(prediction, label.to(self.device))
            train_loss += loss_out.data
            

            # apply backpropagation if indicated
            if(backpropagation):
                loss_out.backward()
                backprop_counter += 1
                # counter makes sure only every X samples backpropagation is
                # excluded (resembling a training batch)
                if(backprop_counter % backprop_every == 0):
                    self.optimizer.step()
                    # print_grads(self.model)
                    self.optimizer.zero_grad()

            # transforms prediction tensor into index of position with highest
            # value
            label_prediction = torch.argmax(prediction, dim=1).item()
            label_groundtruth = label

            # store patient information for potential later analysis
            data_obj.add_patient(
                label_groundtruth,
                path_full[0],
                label_prediction,
                F.softmax(prediction, dim=1).detach(),
                loss_out.detach(),

                )

            # store predictions accordingly in confusion matrix
            if(label_prediction == label_groundtruth):
                corrects += 1
            confusion_matrix[label_groundtruth, label_prediction] += int(1)

            # print('----- loss: {:.3f}, gt: {} , pred: {}, prob: {}'.format(loss_out, label_groundtruth, label_prediction, prediction.detach().cpu().numpy()))

        samples = len(self.dataloaders[split])
        train_loss /= samples

        accuracy = corrects / samples

        print('- ep: {}/{}, loss: {:.3f}, acc: {:.3f}, {}s, {}'.format(
            epoch + 1, self.epochs, train_loss.cpu().numpy(),
            accuracy, int(time.time() - time_pre_epoch), split))

        return train_loss, accuracy, confusion_matrix, data_obj.return_data()
    

        


class DataMatrix():
    '''DataMatrix contains all information about patient classification for later storage.
    Data is stored within a dictionary:

    self.data_dict[true entity] contains another dictionary with all patient paths for
                                the patients of one entity (e.g. AML-PML-RARA, SCD, ...)

    --> In this dictionary, the paths form the keys to all the data of that patient
        and it's classification, stored as a tuple:

        - attention_raw:    attention for all single cell images before softmax transform
        - attention:        attention after softmax transform
        - prediction:       Numeric position of predicted label in the prediction vector
        - prediction_vector:Prediction vector containing the softmax-transformed activations
                            of the last AMiL layer
        - loss:             Loss for that patients' classification
        - out_features:     Aggregated bag feature vectors after attention calculation and
                            softmax transform. '''

    def __init__(self):
        self.data_dict = dict()

    def add_patient(
            self,
            entity,
            path_full,
            prediction,
            prediction_vector,
            loss,
            ):
        '''Add a new patient into the data dictionary. Enter all the data packed into a tuple into the dictionary as:
        self.data_dict[entity][path_full] = (attention_raw, attention, prediction, prediction_vector, loss, out_features)

        accepts:
        - entity: true patient label
        - path_full: path to patient folder
        - attention_raw: attention before softmax transform
        - attention: attention after softmax transform
        - prediction: numeric bag label
        - prediction_vector: output activations of AMiL model
        - loss: loss calculated from output actitions
        - out_features: bag features after attention calculation and matrix multiplication

        returns: Nothing
        '''

        if not (entity in self.data_dict):
            self.data_dict[entity] = dict()
        self.data_dict[entity][path_full] = (
            prediction,
            prediction_vector.data.cpu().numpy()[0],
            float(
                loss.data.cpu()),
            )

    def return_data(self):
        return self.data_dict
    
    def clear(self):
        self.data_dict = dict()  # Reset dictionary
