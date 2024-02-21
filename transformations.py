# This Python file uses the following encoding: utf-8
"""
Created on Mon Feb 10 12:11:45 2020

@author: umaer
"""

# Import necessary libraries
import cv2
import torch
import numpy as np

# Define a class to rescale images to a specified size
class Rescale(object):
    def __init__(self, output_size):
        # Ensure the output_size is either an int (for square images) or a tuple (for rectangular images)
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size  # Set the desired output size
        
    def __call__(self, sample):
        # Extract image, target, and demographic information from the sample
        img, target, demographic = sample['image'], sample['target'], sample['demographic']
        # If output_size is an int, resize the image to a square of that size
        if isinstance(self.output_size, int):  
            img_scaled = cv2.resize(img, (self.output_size, self.output_size))
        else:  # If output_size is a tuple, resize the image to that size (width, height)
            img_scaled = cv2.resize(img, self.output_size)

        # Return the transformed sample with the scaled image and original target and demographic data
        return {'image': img_scaled, 'target': target, 'demographic': demographic}
        

# Define a class to normalize image pixel values
class Normalize(object):
    def __call__(self, sample):
        # Extract image, target, and demographic information from the sample
        img, target, demographic = sample['image'], sample['target'], sample['demographic']
        
        # Normalize the image by dividing pixel values by 255 (to scale between 0 and 1)
        img_normalized = img / 255
        
        # Return the transformed sample with the normalized image and original target and demographic data
        return {'image': img_normalized, 'target': target, 'demographic': demographic}
    

# Define a class to convert numpy arrays to PyTorch tensors
class ToTensor(object):
    def __call__(self, sample):
        # Extract image, target, and demographic information from the sample
        image, target, demographic = sample['image'], sample['target'], sample['demographic']
        
        # If the image is grayscale (2D array), add an additional dimension to make it 3D
        if len(image.shape) < 3:
            image = np.expand_dims(image, 2)

        # Rearrange the axes of the image from HWC to CHW format expected by PyTorch
        image = image.transpose((2, 0, 1))
        
        # Convert the image, target, and demographic data from numpy arrays to PyTorch tensors
        return {'image': torch.from_numpy(image),
                'target': torch.tensor(target),
                'demographic': torch.tensor(demographic)}
