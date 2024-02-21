# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:54:08 2020

@author: umaer
"""

# Import necessary libraries
from torch.utils.data import Dataset, DataLoader, random_split
from imgaug import augmenters as iaa
import numpy as np
import pandas as pd
import glob
import cv2
import random
import torch

# Define the PSGDataset class that inherits from Dataset
class PSGDataset(Dataset):
    # Initialize the dataset with configuration and optional transformation
    def __init__(self, config, transform=None):
        # Store configuration parameters
        self.img_dir = config.img_dir
        self.train_label_dir = config.train_label_dir
        self.test_label_dir = config.test_label_dir
        self.label = config.label
        self.n_images = config.n_images
        self.n_channels = config.n_channels
        self.predict_mode = config.predict_mode
        self.train_fraction = config.train_fraction
        self.val_fraction = config.val_fraction
        self.split_mode = config.split_mode
        self.batch_size = config.batch_size
        self.imbalance = config.imbalance
        self.augmentation = config.augmentation
        self.transform = transform
        # Load demographic data
        self.train_demographics = self.get_demographics(self.train_label_dir, self.train_label_dir)
        self.test_demographics = self.get_demographics(self.test_label_dir, self.train_label_dir)
        # Generate file paths for images
        self.train_img_paths = self.get_files_list(self.train_demographics)
        self.test_img_paths = self.get_files_list(self.test_demographics) 
        # Create samples from the file paths
        self.train_samples = self.get_samples(self.train_img_paths, self.train_demographics)
        self.test_samples = self.get_samples(self.test_img_paths, self.test_demographics)
        # Split datasets into training, validation, and test loaders
        self.train_loader, self.val_loader, self.test_loader = self.dataset_split()
                            
    # Return the length of the dataset
    def __len__(self):
        return len(self.samples)
    
    # Get a specific item from the dataset by index
    def __getitem__(self, idx):                    
        return self.samples[idx]
    
    # Load and preprocess demographic data from Excel files
    def get_demographics(self, excel_path, excel_path2):
        # Columns to consider for dropping rows with NaN values
        drop_rows = ['sex', 'age', 'bmi', 'snore', 'observed', 'tired', 
                     'hypertension', 'sex_cat', 'age_cat', 'bmi_cat', 'asq']
        # Load Excel data and preprocess
        demographics = pd.read_excel(excel_path)
        demographics = demographics.dropna(subset=drop_rows)
        df_norm = pd.read_excel(excel_path2)
        df_norm = df_norm.dropna(subset=drop_rows)
        # Normalize and encode demographic data
        demographics['sex'] = (demographics['sex'] == 'M').astype(np.int)
        demographics['age'] = (demographics['age']-min(df_norm['age']))/(max(df_norm['age'])-min(df_norm['age']))
        demographics['bmi'] = (demographics['bmi']-min(df_norm['bmi']))/(max(df_norm['bmi'])-min(df_norm['bmi']))        
        return demographics
        
    # Generate a list of file paths for images based on demographics
    def get_files_list(self, demographics):
        total_channels = 16
        # Filter out non-target files and organize paths by subject
        files = [x for x in glob.glob(self.img_dir + '*.png') if not x.endswith('geometry.png')]
        files = [file for file in files if file.split('\\')[1][0:9] in demographics['s_code'].values]
        # Select images based on the configuration
        if self.n_images == 1:
            img_paths = files[::total_channels]
        # Other configurations for different numbers of images
        # This block organizes the image paths based on the desired number of images per subject
        return img_paths
    
    # Create samples from image paths and demographic data
    def get_samples(self, img_paths, demographics):
        samples = []
        count = 0
        for img_path in img_paths:
            # Extract subject code and load data
            subject_code = img_path.split('\\')[1][0:9] if self.n_images == 1 else img_path[0].split('\\')[1][0:9]
            data = self.get_subject_data(img_path)
            target, predictor = self.get_target(subject_code, demographics)
            df_subject = demographics[demographics['s_code'] == subject_code]
            # Extract and organize demographic data for the subject
            demographic = [df_subject['sex'].values[0], df_subject['age'].values[0], 
                           df_subject['bmi'].values[0], df_subject['snore'].values[0], 
                           df_subject['observed'].values[0], df_subject['tired'].values[0], 
                           df_subject['hypertension'].values[0], df_subject['sex_cat'].values[0],
                           df_subject['age_cat'].values[0], df_subject['bmi_cat'].values[0]]
            # Create sample dictionary
            sample = {'image': data, 'target': target, 'demographic': demographic}
            # Apply transformation if any
            if self.transform:
                sample = self.transform(sample)
            # Add additional information to the sample
            sample['subject_code'] = subject_code
            sample[self.label] = predictor
            # Append the sample to the list
            samples.append(sample)
            count += 1
            # Progress logging
            if count % 100 == 0:
                print('{} subjects loaded of {}'.format(count, len(img_paths)))      
        return samples
    
    # Convert BGR image format to RGB
    def bgr_to_rgb(self, img):
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        return img
    
    # Load subject data from image paths
    def get_subject_data(self, img_path):
        # Load a single image or multiple images per subject based on configuration
        if self.n_images == 1:
            data = self.bgr_to_rgb(cv2.imread(img_path))
        else:
            data = np.empty([845,1024,self.n_channels])
            channel = 0
            for image in img_path: 
                if image.endswith('RGB.png'):
                    data[:,:,channel:channel+3] = self.bgr_to_rgb(cv2.imread(image))
                    channel += 3
                else:
                    data[:,:,channel] = cv2.imread(image)[:,:,0]
                    channel += 1
        return data 
    
    # Determine the target value for a subject based on demographic data
    def get_target(self, subject_code, demographics):
        # Extract target information from demographics
        dem = demographics.loc[demographics['s_code'] == subject_code][self.label].values[0]
        # Determine the target based on the label configuration
        # This block handles various types of targets and prediction modes
        if self.label == 'sex':
            target = 1 if dem == 1 else 0
        elif self.label == 'age':
            target = 80 if dem == '>=79' else dem
        elif self.label == 'bmi':
            target = dem
        elif self.label == 'hypopnea_fraction' or self.label == 'obs_dur' or \
             self.label == 'hyp_dur' or self.label == 'events_dur' or \
             self.label == 'rem_ahi' or self.label == 'rem_nrem_ratio':
            target = dem
        elif self.label == 'ahi' or self.label == 'ahi2' or self.label == 'odi' or \
             self.label == 'hb' or self.label == 'drop' or \
             self.label == 'duration' or self.label == 'baseline' or \
             self.label == 'below_90_frac':
            if self.predict_mode == 'two-class':
                target = 1 if dem >= 15 else 0
            elif self.predict_mode == 'multi-class':
                if dem < 5:
                    target = 0
                elif dem >= 5 and dem < 15:
                    target = 1
                elif dem >= 15 and dem < 30:
                    target = 2
                else:
                    target = 3
            else:
                target = dem
        return target, dem
    
    # Split the dataset into training, validation, and test sets
    def dataset_split(self):      
        # Calculate sizes for training and validation sets
        train_size = int(self.train_fraction * len(self.train_samples))
        val_size = len(self.train_samples) - train_size
        # Split the dataset based on the split mode
        if self.split_mode == 'random':
            train_dataset, val_dataset = random_split(self.train_samples, [train_size, val_size])
        # Other split modes could be defined here
        # Handle class imbalance if needed
        if self.imbalance == 'sampling':
            train_dataset = self.random_sampling(train_dataset, self.predict_mode)
        # Apply data augmentation if enabled
        if self.augmentation:
            train_dataset = self.augment_data(train_dataset, 2000)
        # Create DataLoader instances for training, validation, and test sets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)    
        val_loader = DataLoader(val_dataset, batch_size=int(self.batch_size), drop_last=True)    
        test_loader = DataLoader(self.test_samples, batch_size=int(self.batch_size), drop_last=True)
        return train_loader, val_loader, test_loader
    
    # Apply random sampling to address class imbalance
    def random_sampling(self, dataset, predict_mode):    
        # Sort dataset by target and apply sampling strategies
        # This block adjusts the dataset to mitigate class imbalance issues
        dataset = sorted(dataset, key = lambda data: data['target'])
        targets = [float(data['target']) for data in dataset]
        targets = np.array(targets)
        
        if predict_mode == 'regression':
            first_target = np.where(targets > 20)[0][0]
            random_samples = random.choices(dataset[first_target:len(targets)], k=1500)
            dataset += random_samples
        else:
            _, counts = np.unique(targets, return_counts=True)
            max_val = counts.max()
            class_idx = np.where(np.roll(targets,1) != targets)[0].tolist()
            class_idx.append(len(targets))
            
            for i_class in range(len(counts)):
                diff = max_val - counts[i_class]
                dataset += random.choices(dataset[class_idx[i_class]:
                                                        class_idx[i_class+1]-1], k=diff)
        return dataset
    
    # Split the dataset while keeping the same proportion of classes in both sets
    def same_split(self, dataset, train_length):
        # Shuffle and split the dataset into training and validation sets
        random.seed(42)
        random.shuffle(dataset)
        train_data = dataset[:train_length]
        val_data = dataset[train_length:]
        return train_data, val_data
    
    # Apply data augmentation to the dataset
    def augment_data(self, dataset, n_images):
        # Apply a sequence of image augmentations to increase dataset size and variability
        random_samples = random.choices(dataset, k=n_images)
        images = [np.moveaxis(data['image'].numpy(),0,-1) for data in random_samples]
        seq = iaa.Sequential([
            iaa.Fliplr(1),
            iaa.Flipud(0.5),
            iaa.Sometimes(
                0.3, iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-2, 2))
            )
        ], random_order=True)
        images = seq(images=images)

        for i in range(len(random_samples)):
            random_samples[i]['image'] = torch.tensor(np.moveaxis(images[i], -1, 0).astype(np.float64))
        dataset += random_samples
        return dataset
