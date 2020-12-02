###########################################################################
#CGNet: A Light-weight Context Guided Network for Semantic Segmentation
#Paper-Link: https://arxiv.org/pdf/1811.08201.pdf
###########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from climatenet.utils.data import ClimateDataset, ClimateDatasetLabeled
from climatenet.modules import *
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_cm, get_iou_perClass
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class CGNet():

    def __init__(self, config: dict):

        self.fields = config['model']['fields']
        self.lr = config['model']['lr']

        self.network = cgnet_module(classes=3, channels=len(self.fields)).cuda()
        self.optimizer = Adam(self.network.parameters(), lr=1e-3)
        
    def train(self, dataset: ClimateDatasetLabeled, loss: str, epochs: int = 1):
        self.network.train()
        loader = DataLoader(dataset, batch_size=8)
        for epoch in range(1, epochs+1):

            print(f'Epoch {epoch}:')
            epoch_loader = tqdm(loader)
            aggregate_cm = np.zeros((3,3))

            for inputs, labels in epoch_loader:
        
                # Push data on GPU and pass forward
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = torch.softmax(self.network(inputs), 1)

                # Update training CM
                predictions = torch.max(outputs, 1)[1]
                aggregate_cm += get_cm(predictions, labels, 3)

                # Pass backward
                loss = jaccard_loss(outputs, labels)
                epoch_loader.set_description(f'Loss: {loss.item()}')
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() 

            print('Epoch stats:')
            print(aggregate_cm)
            ious = get_iou_perClass(aggregate_cm)
            print('IOUs: ', ious, ', mean: ', ious.mean())

    def predict(self, dataset: ClimateDataset):
        self.network.eval()
        loader = DataLoader(dataset, batch_size=1)
        epoch_loader = tqdm(loader)

        preds = []

        for inputs in epoch_loader:
        
            # Push data on GPU and pass forward
            inputs = inputs.cuda().squeeze()
            with torch.no_grad():
                outputs = torch.softmax(self.network(inputs), 1)
            batch_preds = torch.max(outputs, 1)[1]

            preds.append(batch_preds)

        return preds


                


    def evaluate(self, dataset: ClimateDatasetLabeled):
        pass

    def save_weights(self, path: str):
        pass

class cgnet_module(nn.Module):
    """
    This class defines the proposed Context Guided Network (CGNet) in this work.
    """
    def __init__(self, classes=19, channels=4, M= 3, N= 21, dropout_flag = False):
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
            print("have droput layer")
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

        # upsample segmenation map ---> the input image size
        out = F.interpolate(classifier, input.size()[2:], mode='bilinear',align_corners = False)   #Upsample score map, factor=8
        return out
      
   