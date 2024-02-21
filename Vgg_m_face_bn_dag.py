# This Python file uses the following encoding: utf-8
"""
Created on Thu Jan 16 16:13:29 2020

@author: umaer
"""

# Import PyTorch library for neural network construction
import torch
import torch.nn as nn

# Define a class for the VGG Face model, inheriting from PyTorch's nn.Module
class Vgg_face_dag(nn.Module):

    def __init__(self):
        # Initialize the parent class (nn.Module)
        super(Vgg_face_dag, self).__init__()
        # Metadata for preprocessing images. Includes mean, std, and imageSize for normalization and resizing.
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        
        # Define the architecture of the VGG Face model with convolutional, ReLU activation, and max-pooling layers.
        # The network architecture uses multiple blocks of convolutional layers followed by a max-pooling layer to extract features from the input image.
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        
        # The architecture continues with increasing depth, doubling the number of filters with each stage up to 512.
        # This pattern is typical for VGG architectures, aiming to capture complex features at various scales.
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        # Additional convolutional and pooling layers are defined similarly, following the VGG architecture pattern.
        
        # Final layers of the network include fully connected layers to classify the features extracted by the convolutional layers.
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)  # First fully connected layer
        self.relu6 = nn.ReLU(inplace=True)  # Activation function
        self.dropout6 = nn.Dropout(p=0.5)  # Dropout for regularization
        # Similar pattern for the next fully connected layers
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)  # Final layer for classification

    def forward(self, x0):
        # Define the forward pass through the network, applying each layer to the input sequentially.
        # The sequence includes convolutions, activations, and pooling, following the architecture defined in __init__.
        # The output of the final pooling layer is flattened before being passed through the fully connected layers.
        # The method returns the output of the final layer, which can be used for classification.
        x1 = self.conv1_1(x0)
        # Additional operations are applied similarly, following the architecture pattern.
        
        # The output of the last fully connected layer (x38) is returned, representing the model's predictions.
        return x38

def vgg_face_dag(weights_path=None, **kwargs):
    # Instantiate the Vgg_face_dag model
    model = Vgg_face_dag()
    # If a path to pretrained weights is provided, load these weights into the model
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    # Return the model instance, ready for further use or training
    return model
