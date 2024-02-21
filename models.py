# This Python file uses the following encoding: utf-8
"""
Created on Tue Feb 11 13:40:06 2020

@author: umaer
"""

# Import necessary libraries from PyTorch for model development
import torch
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn.functional as F
# Assuming Vgg_m_face_bn_dag.py is a custom file containing the Vgg_face_dag class for facial recognition tasks
from Vgg_m_face_bn_dag import Vgg_face_dag
from torch.nn import (Sequential, Conv2d, Linear, ReLU, Dropout, Module, 
                      MaxPool2d, BatchNorm2d, L1Loss, CrossEntropyLoss, 
                      BCEWithLogitsLoss, SmoothL1Loss, MSELoss)

# Define a class for a custom transfer learning network based on ResNet architecture
class TransferNetwork(ResNet):
    def __init__(self, config):
        # Initialize the base ResNet model with BasicBlock and a custom layer structure [2, 2, 2, 2] similar to ResNet18
        super(TransferNetwork, self).__init__(block=BasicBlock, layers=[2,2,2,2])
        # Configuration settings for the model, including number of input channels and prediction mode
        self.input_dim = config.n_channels
        self.label = config.label
        self.predict_mode = config.predict_mode
        self.imbalance = config.imbalance
        # Customize the first convolutional layer to adapt to input dimensionality
        self.conv1 = Conv2d(self.input_dim, out_channels=64, kernel_size=(7,7), 
                            stride=(2,2), padding=(3,3), bias=False)
        # Define a sequential model for the fully connected (fc) layer part with dropout for regularization
        self.fc = Sequential(Linear(512, 512), ReLU(), Dropout(0.3),
                             Linear(512, 128), ReLU(), Dropout(0.3))
                             
        # Configure the output layer and loss function based on the task (binary classification, multi-class classification, or regression)
        if self.label == 'sex' or (self.label == 'ahi' and self.predict_mode == 'two-class'):
            self.fc.output = Linear(128, 2)  # Output layer for binary classification
            # Use weighted BCEWithLogitsLoss for imbalanced datasets or standard CrossEntropyLoss otherwise
            self.loss = BCEWithLogitsLoss(pos_weight=torch.tensor(1.8)) if self.imbalance == 'weights' else CrossEntropyLoss()
        elif self.label == 'ahi' and self.predict_mode == 'multi-class':
            self.fc.output = Linear(128, 4)  # Output layer for multi-class classification
            # Use weighted CrossEntropyLoss for imbalanced datasets or standard CrossEntropyLoss otherwise
            self.loss = CrossEntropyLoss(weight=torch.tensor([1.4, 1.0, 2.6, 5.0])) if self.imbalance == 'weights' else CrossEntropyLoss()
        else:
            self.fc.output = Linear(128, 1)  # Output layer for regression
            self.loss = MSELoss()  # Use Mean Squared Error Loss for regression tasks

    def forward(self, img, demographics):
        # Forward pass of the network: process input image through conv layers, activation functions, pooling, and fully connected layers
        img = self.conv1(img)
        img = self.bn1(img)
        img = self.relu(img)
        img = self.maxpool(img)
        img = self.layer1(img)
        img = self.layer2(img)
        img = self.layer3(img)
        img = self.layer4(img)
        img = self.avgpool(img)
        img = torch.flatten(img, 1)  # Flatten the output for the fully connected layer
        img = self.fc(img)  # Pass through the fully connected layer
        return img

# Define a ResNet34 class that inherits from torchvision's ResNet class
class ResNet34(ResNet):
    def __init__(self, config):
        # Initialize the ResNet model with a configuration typical for ResNet34
        super(ResNet34, self).__init__(block=BasicBlock, layers=[3,4,6,3])
        # Set up model configuration using the provided settings
        self.input_dim = config.n_channels  # Define the number of input channels
        self.label = config.label  # Specify the label for the task (e.g., 'sex', 'ahi')
        self.predict_mode = config.predict_mode  # Define the prediction mode (e.g., 'multi-class')
        self.imbalance = config.imbalance  # Handle class imbalance if applicable
        # Redefine the first convolutional layer to match input dimensions
        self.conv1 = Conv2d(self.input_dim, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        # Define a fully connected layer sequence for processing the features
        self.fc = Sequential(Linear(512, 512), ReLU(), Dropout(0.3), Linear(512, 128), ReLU(), Dropout(0.5))
        
        # Adjust the final layer and loss function according to the task
        if self.label == 'sex' or (self.label == 'ahi' and self.predict_mode == 'two-class'):
            self.fc.output = Linear(128, 1)  # Binary output
            # Use BCEWithLogitsLoss with or without class weights for imbalanced datasets
            self.loss = BCEWithLogitsLoss(pos_weight=torch.tensor(1.8)) if self.imbalance == 'weights' else BCEWithLogitsLoss()
        elif self.label == 'ahi' and self.predict_mode == 'multi-class':
            self.fc.output = Linear(128, 4)  # Output layer for 4 classes
            # Choose weighted or standard CrossEntropyLoss based on class imbalance
            self.loss = CrossEntropyLoss(weight=torch.tensor([1.4, 1.0, 2.6, 5.0])) if self.imbalance == 'weights' else CrossEntropyLoss()
        else:
            self.fc.output = Linear(128, 1)  # Single output for regression tasks
            self.loss = MSELoss()  # Use Mean Squared Error Loss for regression

# Define a class for a custom VGG-Face model
class VggFace(Vgg_face_dag):
    def __init__(self, config):
        super(VggFace, self).__init__()  # Initialize the parent class
        # Configure the model based on provided settings
        self.input_dim = config.n_channels
        self.label = config.label
        self.predict_mode = config.predict_mode
        self.imbalance = config.imbalance
        self.weights_path = config.weights_path  # Path to pretrained weights, if applicable
        state_dict = torch.load(self.weights_path)  # Load the pretrained weights
        self.load_state_dict(state_dict)  # Apply the weights to the model
        
        # Freeze the parameters (weights) of the model to prevent them from being updated during training
        for param in self.parameters():
            param.requires_grad = False
        
        # Adjust the model for specific tasks (binary classification, multi-class classification, or regression)
        if self.label == 'sex' or (self.label == 'ahi' and self.predict_mode == 'two-class'):
            self.fc8 = Linear(4096, 1)  # Adjust the output layer for binary tasks
            # Choose BCEWithLogitsLoss with or without class weights for imbalanced datasets
            self.loss = BCEWithLogitsLoss(pos_weight=torch.tensor(2.0)) if self.imbalance == 'weights' else BCEWithLogitsLoss()
        elif self.label == 'ahi' and self.predict_mode == 'multi-class':
            self.fc8.output = Linear(256, 4)  # Adjust the output layer for multi-class tasks
            # Choose weighted or standard CrossEntropyLoss based on class imbalance
            self.loss = CrossEntropyLoss(weight=torch.tensor([1.4, 1.0, 2.6, 5.0])) if self.imbalance == 'weights' else CrossEntropyLoss()        
        else:
            self.fc8.output = Linear(256, 1)  # Single output for regression tasks
            self.loss = MSELoss()  # Use Mean Squared Error Loss for regression

# CustomNetwork class demonstrates how to create a neural network from scratch using PyTorch's basic building blocks
class CustomNetwork(Module):
    def __init__(self, config):
        super(CustomNetwork, self).__init__()
        # Initial configuration based on the provided settings
        self.input_dim = config.n_channels
        self.label = config.label        
        self.predict_mode = config.predict_mode
        self.imbalance = config.imbalance
        
        # Define the convolutional layers with increasing number of filters and kernel sizes optimized for feature extraction
        self.conv1 = Conv2d(in_channels=self.input_dim, out_channels=16, kernel_size=7)
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv4 = Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        
        # Pooling layer to reduce dimensionality and increase receptive field
        self.pool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers to map the features to the output
        self.fc1 = Linear(in_features=128*11*11, out_features=1024)
        self.fc2 = Linear(in_features=1024, out_features=128)
        
        # Adjust the output layer based on the task
        if self.label == 'ahi' and self.predict_mode == 'multi-class':
            self.output = Linear(in_features=128, out_features=4)  # Multi-class classification
        else:
            self.output = Linear(in_features=128, out_features=1)  # Binary classification or regression
        
        # Dropout layers to prevent overfitting
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.3)
        self.dropout3 = Dropout(0.5)
        
        # Batch normalization layers to stabilize and speed up training
        self.batchnorm1 = BatchNorm2d(16)
        self.batchnorm2 = BatchNorm2d(32)
        self.batchnorm3 = BatchNorm2d(64)
        self.batchnorm4 = BatchNorm2d(128)
        
        # Define the loss function based on the task and imbalance handling
        if self.label == 'sex' or (self.label == 'ahi' and self.predict_mode == 'two-class'):
            self.loss = BCEWithLogitsLoss(pos_weight=torch.tensor(3.0)) if self.imbalance == 'weights' else BCEWithLogitsLoss()
        elif self.label == 'ahi' and self.predict_mode == 'multi-class':
            self.loss = CrossEntropyLoss(weight=torch.tensor([1.4, 1.0, 2.6, 5.0])) if self.imbalance == 'weights' else CrossEntropyLoss()
        else:
            self.loss = L1Loss()  # L1 Loss for regression tasks

    def forward(self, x):
        # Forward pass: Apply convolutional layers, activation functions, pooling, and fully connected layers
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.reshape(-1, 128*11*11)  # Flatten the output for the fully connected layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.output(x)  # Final output layer

        return output
