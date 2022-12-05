import json
import torch
import numpy as np

class Config():
    '''
    Abstracts over a model configuration.
    While it currently does not offer any advantages over working with a simple dict,
    it makes it possible to simply add functionality concerning configurations:
    - Checking the validity of a configuration file
    - Automatically loading and saving configuration files

    Parameters
    ----------
    path : str
        The path to the json with the config

    Attributes
    ----------
    architecture : str
        Stores the model architecture type. Currently ignored (only have CGNet), but can be used in the future
    lr : dict
        The learning rate used to train the model
    fields : [str]
        A dictionary mapping from variable names to normalisation statistics
    description : str
        Stores an uninterpreted description string for the model. Put anything you want here.
    '''

    def __init__(self, path: str):
        self.config_dict = json.load(open(path))

        # TODO: Check structure

        self.architecture = self.config_dict['architecture']
        self.lr = self.config_dict['lr']
        self.seed = self.config_dict['seed']
        self.train_batch_size = self.config_dict['train_batch_size']
        self.pred_batch_size = self.config_dict['pred_batch_size']
        self.epochs = self.config_dict['epochs']
        self.fields = self.config_dict['fields']
        self.labels = self.config_dict['labels']
        self.description = self.config_dict['description']
        self.loss = self.config_dict['loss']

        if 'weights' in self.config_dict: self.weights = self.config_dict['weights']  # TODO: raise exception if None
        self.save_epochs = self.config_dict.get('save_epochs', False) 
        # Make reproducible
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def save(self, save_path: str):
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.config_dict, f, ensure_ascii=False, indent=4)
        
