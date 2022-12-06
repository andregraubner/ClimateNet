###########################################################################
#CGNet: A Light-weight Context Guided Network for Semantic Segmentation
#Paper-Link: https://arxiv.org/pdf/1811.08201.pdf
###########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from climatenet.modules import *
from climatenet.utils.data import ClimateDataset, ClimateDatasetLabeled
from climatenet.utils.losses import loss_function
from climatenet.utils.metrics import get_cm, get_iou_perClass, get_dice_perClass, get_confusion_metrics
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import xarray as xr
import pandas as pd
from climatenet.utils.utils import Config
from os import path
import pathlib

class CGNet():
    '''
    The high-level CGNet class. 
    This allows training and running CGNet without interacting with PyTorch code.
    If you are looking for a higher degree of control over the training and inference,
    we suggest you directly use the CGNetModule class, which is a PyTorch nn.Module.

    Parameters
    ----------
    config : Config
        The model configuration.
    model_path : str
        Path to load the model and config from.

    Attributes
    ----------
    config : dict
        Stores the model config
    network : CGNetModule
        Stores the actual model (nn.Module)
    optimizer : torch.optim.Optimizer
        Stores the optimizer we use for training the model
    '''

    def __init__(self, config: Config = None, model_path: str = None):
    
        if config is not None and model_path is not None:
            raise ValueError('''Config and weight path set at the same time. 
            Pass a config if you want to create a new model, 
            and a weight_path if you want to load an existing model.''')
        if config is not None:
            # Create new model
            self.config = config
            self.network = CGNetModule(classes=len(self.config.labels), channels=len(list(self.config.fields)))
        elif model_path is not None:
            # Load model
            self.config = Config(path.join(model_path, 'config.json'))
            self.network = CGNetModule(classes=len(self.config.labels), channels=len(list(self.config.fields)))
            self.network.load_state_dict(torch.load(path.join(model_path, 'weights.pth')))
        else:
            raise ValueError('''You need to specify either a config or a model path.''')

        self.optimizer = Adam(self.network.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=1, threshold=0.002, verbose=True)        
        
    def train(self, train_dataset: ClimateDatasetLabeled, val_dataset: ClimateDatasetLabeled):
        '''Train the network on the train dataset for the given amount of epochs, and validate it
        at each epoch on the validation dataset.'''
        self.network.train()
        
        # Initialize training history
        history = pd.DataFrame(columns=['minibatch_loss', 'epoch_avg_loss', 'epoch_val_loss', 'learning_rate', 'evaluation_loss',\
                                        'training_confusion_matrix', 'validation_confusion_matrix', 'evaluation_confusion_matrix',\
                                        'train_ious', 'train_dices', 'train_precision', 'train_recall', 'train_specificity', 'train_sensitivity',\
                                        'val_ious', 'val_dices', 'val_precision', 'val_recall', 'val_specificity', 'val_sensitivity',\
                                        'test_ious', 'test_dices', 'test_precision', 'test_recall', 'test_specificity', 'test_sensitivity'])

        # Push model to GPU if available
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.network.to(device)

        collate = ClimateDatasetLabeled.collate
        loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size, collate_fn=collate, num_workers=0, shuffle=True)

        # Prepare scheduler
        best_val_loss = float("inf")
        no_improvement_counter = 0

        # Loop over epochs
        for epoch in range(1, self.config.epochs+1):

            print(f'\n================ Training epoch #{epoch} ({device}) ================')
            epoch_loader = tqdm(loader, leave=True)
            train_aggregate_cm = np.zeros((3,3))
            num_minibatches = len(loader)
            epoch_loss = 0.
            train_loss = 0.

            for features, labels in epoch_loader:
        
                # Move dataset to GPU if available
                features = torch.tensor(features.values)
                labels = torch.tensor(labels.values)

                features = features.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = torch.softmax(self.network(features), 1)

                # Update training confusion matrix
                predictions = torch.max(outputs, 1)[1]
                train_aggregate_cm += get_cm(predictions, labels, 3)

                # Backward pass
                train_loss = loss_function(outputs, labels, config_loss=self.config.loss)
                epoch_loader.set_description(f'Loss: {train_loss.item():.5f} ({self.config.loss}) | LR: {self.optimizer.param_groups[0]["lr"]}')
                
                epoch_loss += train_loss
                history = history.append({'minibatch_loss': train_loss.item()}, ignore_index=True)

                train_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() 

            # Compute and track training hisotry
            epoch_loss /= num_minibatches
            training_confusion_matrix = 100*train_aggregate_cm/np.sum(train_aggregate_cm)
            train_ious = get_iou_perClass(train_aggregate_cm)
            train_dices = get_dice_perClass(train_aggregate_cm)
            t_precision, t_recall, t_specificity, t_sensitivity = get_confusion_metrics(train_aggregate_cm)

            history = history.append({'epoch_avg_loss': epoch_loss.item(), 'learning_rate': self.optimizer.param_groups[0]["lr"], \
                                    'training_confusion_matrix': np.array(training_confusion_matrix),\
                                    'train_ious': train_ious, 'train_dices': train_dices,\
                                    'train_precision': t_precision, 'train_recall': t_recall,\
                                    'train_specificity': t_specificity, 'train_sensitivity': t_sensitivity}, ignore_index=True)            

            # Training stats reporting
            print(f'\nTraining loss: {epoch_loss:.5f} ({self.config.loss}) ')
            print('Classes:      [    BG         TCs        ARs   ]')
            print('IoUs:        ', train_ious, ' | Mean: ', train_ious.mean())
            print('Dice score:  ', train_dices, ' | Mean: ', train_dices.mean())            
            print("Precision:   ", t_precision)
            print("Recall:      ", t_recall)
            print("Specificity: ", t_specificity)
            print("Sensitivity: ", t_sensitivity)
            print(np.array_str(np.around(training_confusion_matrix, decimals=3), precision=3))

            # Compute and track validation history
            val_loss, val_aggregate_cm, val_ious, val_dices = self.validate(val_dataset)

            validation_confusion_matrix = 100*val_aggregate_cm/np.sum(val_aggregate_cm)
            v_precision, v_recall, v_specificity, v_sensitivity = get_confusion_metrics(val_aggregate_cm)

            history = history.append({'epoch_val_loss': val_loss, 'learning_rate': self.optimizer.param_groups[0]["lr"],\
                                    'validation_confusion_matrix': np.array(validation_confusion_matrix),\
                                    'val_ious': val_ious, 'val_dices': val_dices,\
                                    'val_precision': v_precision, 'val_recall': v_recall,\
                                    'val_specificity': v_specificity, 'val_sensitivity': v_sensitivity}, ignore_index=True)   

            # Validation stats reporting
            print(f'\nValidation loss: {val_loss:.5f} ({self.config.loss})')
            print('Classes:      [    BG         TCs        ARs   ]')
            print('IoUs:        ', val_ious, ' | Mean: ', val_ious.mean())
            print('Dice score:  ', val_dices, ' | Mean: ', val_dices.mean())
            print("Precision:   ", v_precision)
            print("Recall:      ", v_recall)
            print("Specificity: ", t_specificity)
            print("Sensitivity: ", t_sensitivity)
            print(np.array_str(np.around(validation_confusion_matrix, decimals=3), precision=3))
            
            self.network.train()

            # Update the learning rate if needed
            self.scheduler.step(val_loss)

            # Check for early termination
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_counter = 0
            else:
                best_val_loss *= 1 - 0.002
                no_improvement_counter += 1

            if no_improvement_counter >= 3:
                break

            # Save model at each epoch if specified in config.json
            #if self.config.save_epochs : 
                #self.save_model(self, self.config.model_path)
                #print("Saving weights from epoch #", str(epoch), "\n")

        # Return training history
        return history
        

    def predict(self, dataset: ClimateDataset, save_dir: str = None):
        '''Make predictions for the given dataset and return them as xr.DataArray'''
        self.network.eval()
        collate = ClimateDataset.collate
        loader = DataLoader(dataset, batch_size=self.config.pred_batch_size, collate_fn=collate)
        epoch_loader = tqdm(loader, leave=True)
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        predictions = []
        for batch in epoch_loader:
            features = torch.tensor(batch.values)
            features = features.to(device)

            with torch.no_grad():
                outputs = torch.softmax(self.network(features), 1)
            preds = torch.max(outputs, 1)[1].cpu().numpy()

            coords = batch.coords
            del coords['variable']
            dims = [dim for dim in batch.dims if dim != "variable"]
            
            predictions.append(xr.DataArray(preds, coords=coords, dims=dims, attrs=batch.attrs))
    
        print(predictions)
        return xr.concat(predictions, dim='time')

    def validate(self, dataset: ClimateDatasetLabeled):
        '''Validate on a dataset and return statistics'''

        self.network.eval()
        collate = ClimateDatasetLabeled.collate
        loader = DataLoader(dataset, batch_size=self.config.pred_batch_size, collate_fn=collate, num_workers=0)

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        print(f'\n---------- Validation ({device}) ----------')

        epoch_loader = tqdm(loader, leave=True)
        aggregate_cm = np.zeros((3,3))
        num_minibatches = len(loader)
        epoch_loss = 0.

        for features, labels in epoch_loader:
        
            features = torch.tensor(features.values)
            labels = torch.tensor(labels.values)

            features = features.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = torch.softmax(self.network(features), 1)
            predictions = torch.max(outputs, 1)[1]
            aggregate_cm += get_cm(predictions, labels, 3)

            val_loss = loss_function(outputs, labels, config_loss=self.config.loss)
            epoch_loss += val_loss.item()

        # Return validation stats:
        epoch_loss /= num_minibatches
        return epoch_loss, aggregate_cm, get_iou_perClass(aggregate_cm), get_dice_perClass(aggregate_cm)

    def evaluate(self, dataset: ClimateDatasetLabeled):
        '''Evaluate on a dataset and return statistics'''
        self.network.eval()

        collate = ClimateDatasetLabeled.collate
        loader = DataLoader(dataset, batch_size=self.config.pred_batch_size, collate_fn=collate, num_workers=0)

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        print(f'\n---------- Evaluation ({device}) ----------')

        epoch_loader = tqdm(loader, leave=True)
        aggregate_cm = np.zeros((3,3))
        epoch_loss = 0.
        num_minibatches = len(loader)

        for features, labels in epoch_loader:
        
            features = torch.tensor(features.values)
            labels = torch.tensor(labels.values)

            features = features.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = torch.softmax(self.network(features), 1)
            predictions = torch.max(outputs, 1)[1]
            aggregate_cm += get_cm(predictions, labels, 3)

            test_loss = loss_function(outputs, labels, config_loss=self.config.loss)
            epoch_loss += val_loss.item()

            epoch_loss += test_loss.item()

        # Compute and track evaluation stats
        epoch_loss /= num_minibatches
        test_confusion_matrix = 100*aggregate_cm/np.sum(aggregate_cm)
        test_precision, test_recall, test_specificity, test_sensitivity = get_confusion_metrics(aggregate_cm)
        test_ious = get_iou_perClass(aggregate_cm)
        test_dices = get_dice_perClass(aggregate_cm)

        history = pd.DataFrame.from_dict({'evaluation_loss': [epoch_loss],
                                  'evaluation_confusion_matrix': [np.array(test_confusion_matrix)],
                                  'test_ious': [test_ious],
                                  'test_dices': [test_dices],
                                  'test_precision': [test_precision],
                                  'test_recall': [test_recall],
                                  'test_specificity': [test_specificity],
                                  'test_sensitivity': [test_sensitivity]})

        # Evaluation stats reporting:
        print(f'\nTest loss: {epoch_loss:.5f} ({self.config.loss})')         
        print('Classes:      [    BG         TCs        ARs   ]')
        print('IoUs:        ', test_ious, ' | Mean: ', test_ious.mean())
        print('Dice score:  ', test_dices, ' | Mean: ', test_dices.mean())
        print("Precision:   ", test_precision)
        print("Recall:      ", test_recall)
        print("Specificity: ", test_specificity)
        print("Sensitivity: ", test_sensitivity)
        print(np.array_str(np.around(test_confusion_matrix, decimals=3), precision=3))
  
        # Return evaluation metrics
        return history

    def save_model(self, save_path: str):
        '''
        Save model weights and config to a directory.
        '''
        # create save_path if it doesn't exist
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 

        # save weights and config
        self.config.save(path.join(save_path, 'config.json'))
        torch.save(self.network.state_dict(), path.join(save_path, 'weights.pth'))

    def load_model(self, model_path: str):
        '''
        Load a model. While this can easily be done using the normal constructor, this might make the code more readable - 
        we instantly see that we're loading a model, and don't have to look at the arguments of the constructor first.
        '''
        self.config = Config(path.join(model_path, 'config.json'))
        self.network = CGNetModule(classes=len(self.config.labels), channels=len(list(self.config.fields)))
        self.network.load_state_dict(torch.load(path.join(model_path, 'weights.pth')))
    
    def print_model(self, dataset: ClimateDatasetLabeled, depth=1):
        """
        Prints a summary of the network using torchinfo (https://github.com/TylerYep/torchinfo).

        args:
            input: Receives the CGNet object to print
        """
        from torchinfo import summary

        # Load data for a forward pass
        collate = ClimateDatasetLabeled.collate
        loader = DataLoader(dataset, batch_size=self.config.train_batch_size, collate_fn=collate, num_workers=0, shuffle=True)

        # Print summary using 'torchinfo'
        summary(self.network, \
                input_size=next(iter(loader))[0].shape, \
                col_names=[\
                "input_size",\
                "output_size",\
                #"kernel_size",\
                "num_params", \
                "mult_adds"],\
                #"trainable"],\
                depth=depth)

class CGNetModule(nn.Module):
    """
    CGNet (Wu et al, 2018: https://arxiv.org/pdf/1811.08201.pdf) implementation.
    This is taken from their implementation, we do not claim credit for this.
    """
    def __init__(self, classes=19, channels=4, M=3, N= 21, dropout_flag = False):
        """
        args:
          classes: number of classes in the dataset. Default is 19 for the cityscapes
          M: the number of blocks in stage 2
          N: the number of blocks in stage 3
        """
        super().__init__()

        self.level1_0 = ConvBNPReLU(channels, 32, 3, 2)      # feature map size divided 2, 1/2
        self.level1_1 = ConvBNPReLU(32, 32, 3, 1)                          
        self.level1_2 = ConvBNPReLU(32, 32, 3, 1)      

        self.sample1 = InputInjection(1)  #down-sample for Input Injection, factor=2
        self.sample2 = InputInjection(2)  #down-sample for Input Injiection, factor=4

        self.b1 = BNPReLU(32 + channels)
        
        #stage 2
        self.level2_0 = ContextGuidedBlock_Down(32 + channels, 64, dilation_rate=2,reduction=8)  
        self.level2 = nn.ModuleList()
        for i in range(0, M-1):
            self.level2.append(ContextGuidedBlock(64 , 64, dilation_rate=2, reduction=8))  #CG block
        self.bn_prelu_2 = BNPReLU(128 + channels)
        
        #stage 3
        self.level3_0 = ContextGuidedBlock_Down(128 + channels, 128, dilation_rate=4, reduction=16) 
        self.level3 = nn.ModuleList()
        for i in range(0, N-1):
            self.level3.append(ContextGuidedBlock(128 , 128, dilation_rate=4, reduction=16)) # CG block
        self.bn_prelu_3 = BNPReLU(256)

        if dropout_flag:
            print("have dropout layer")
            self.classifier = nn.Sequential(nn.Dropout2d(0.1, False),Conv(256, classes, 1, 1))
        else:
            self.classifier = nn.Sequential(Conv(256, classes, 1, 1))

        #init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d')!= -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif classname.find('ConvTranspose2d')!= -1:
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        # stage 1
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        
        # stage 2
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.bn_prelu_2(torch.cat([output1,  output1_0, inp2], 1))

        # stage 3
        output2_0 = self.level3_0(output1_cat) # down-sampled
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))
       
        # classifier
        classifier = self.classifier(output2_cat)

        # upsample segmentation map ---> the input image size
        out = F.interpolate(classifier, input.size()[2:], mode='bilinear',align_corners = False)   #Upsample score map, factor=8
        return out
  
   
