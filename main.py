# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:02:38 2020

@author: umaer
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformations import Rescale, Normalize, ToTensor 
from psg_dataset import PSGDataset
import matplotlib.pyplot as plt
from models import TransferNetwork
from torchvision import transforms
from scipy.stats import pearsonr

# Define a class to handle patient demographics and model training/testing
class PatientDemographics(object):
    def __init__(self, config):
        self.config = config  # Configuration settings passed as a parameter
        # Adjust the number of channels based on the number of images
        self.config.n_channels = 2 * self.config.n_images if (self.config.n_images % 2) == 0 else 3 * self.config.n_images       
        # Initialize the dataset with transformations
        self.dataset = PSGDataset(config=self.config, transform=transforms.Compose([Rescale(224), Normalize(), ToTensor()]))
        self.model = TransferNetwork(config=self.config)  # Initialize the model
        self.criterion = self.model.loss  # Loss function from the model
        # Optimizer for model parameters with learning rate and weight decay from config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device configuration
        self.model.to(self.device)  # Move model to the configured device
        # Initialize variables to track training and validation metrics
        self.loss = 0
        self.train_loss = []
        self.val_loss = []
        self.accuracy = []
        self.targets = []
        self.predictions = []
        self.iteration = 0
        self.iterations = []
        self.current_epoch = 0
            
        self.main()  # Begin the main training/testing loop

    # Method to handle training
    def train(self):
        running_loss = 0.0
        self.model.train()  # Set model to training mode
        for i, batch_train in enumerate(self.dataset.train_loader, 0):
            self.optimizer.zero_grad()  # Clear gradients
            _, _, out, loss = self.predict(batch_train)  # Get predictions and loss
            self.loss = loss
            self.optimize()  # Backpropagate and optimize
            running_loss += self.loss.item()  # Accumulate loss
        mean_loss = running_loss / (i + 1)  # Calculate mean loss for epoch
        self.train_loss.append(mean_loss)  # Store mean loss
        self.iterations.append(self.iteration)  # Track iterations
        self.current_epoch += 1  # Increment epoch count
        print('Epoch:', self.current_epoch)
        print('Train loss:', mean_loss)

    # Method for backpropagation and optimization
    def optimize(self):
        self.loss.backward()  # Backpropagate loss
        self.optimizer.step()  # Update model parameters
        self.iteration += 1  # Increment iteration count

    # Method for validation
    def validate(self):
        val_running_loss = 0.0
        total = 0  # Total number of samples
        correct = 0  # Correct predictions count
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            for j, val_batch in enumerate(self.dataset.val_loader, 0):
                _, val_target, val_out, val_loss = self.predict(val_batch)  # Get predictions and loss for validation batch
                val_running_loss += val_loss.item()  # Accumulate loss
                # Prediction logic based on predict_mode
                if self.config.predict_mode == 'multi-class':
                    predicted = F.softmax(val_out, dim=1)  # Apply softmax for multi-class classification
                    _, predicted = torch.max(predicted.data, 1)  # Get class with highest probability
                else:
                    predicted = torch.sigmoid(val_out)  # Apply sigmoid for binary classification
                    predicted = torch.round(predicted)  # Round to get binary output
                
                total += val_target.size(0)  # Update total sample count
                correct += (predicted == val_target).sum().item()  # Update correct predictions count
            val_mean_loss = val_running_loss / (j + 1)  # Calculate mean validation loss
        
        # Save model checkpoint if validation loss improves
        if not any(x < val_mean_loss for x in self.val_loss):
            self.save_checkpoint()
                
        self.val_loss.append(val_mean_loss)  # Store mean validation loss
        print('Validation loss:', val_mean_loss)
              
        # Calculate and print accuracy for classification tasks
        if self.config.label == 'sex' or (self.config.label == 'ahi' and self.config.predict_mode != 'regression'):
            accuracy = correct / total  # Calculate accuracy
            self.accuracy.append(accuracy)  # Store accuracy
            print('Validation accuracy: %d %%' % (accuracy * 100))
                
    # Method for testing
    def test(self):
        # Reset or initialize variables to store test results
        self.targets = []
        self.predictions = []
        self.images = np.empty([0, self.config.n_channels, 224, 224])
        self.demographics = []
        self.subject_codes = []
        self.target_values = np.empty(0)
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            for k, test_batch in enumerate(self.dataset.test_loader, 0):
                _, test_target, test_out, test_loss = self.predict(test_batch)  # Get predictions and loss for test batch
                # Prediction logic based on label and predict_mode
                if self.config.label == 'ahi' and self.config.predict_mode == 'multi-class':
                    test_out = F.softmax(test_out, dim=1)  # Apply softmax for multi-class classification
                    _, predicted = torch.max(test_out.data, 1)  # Get class with highest probability
                elif self.config.label == 'sex' or (self.config.label == 'ahi' and self.config.predict_mode == 'two-class'):
                    test_out = torch.sigmoid(test_out.data)  # Apply sigmoid for binary classification
                    predicted = torch.round(test_out)  # Round to get binary output
                else:
                    predicted = test_out  # Use raw output for regression tasks                
                # Update lists with test results
                self.targets += test_target.tolist()
                self.predictions += predicted.tolist()
                self.images = np.concatenate((self.images, test_batch['image'].numpy()))  # Concatenate image data
                self.demographics += test_batch['demographic'].tolist()  # Concatenate demographic data
                self.subject_codes += test_batch['subject_code']  # Concatenate subject codes
                self.target_values = np.concatenate((self.target_values, test_batch[self.config.label]))  # Concatenate target values
        self.images = np.moveaxis(self.images, 1, -1)  # Adjust axis for image data
        self.targets = np.asarray(self.targets)  # Convert targets list to array
        self.predictions = np.asarray(self.predictions)  # Convert predictions list to array
        self.df_errors = self.errors_table()  # Generate errors table
        self.performance()

    # Method to generate predictions and calculate loss
    def predict(self, batch):
        data, target, demographic = batch['image'], batch['target'], batch['demographic']  # Unpack batch data
        data = data.to(self.device, dtype=torch.float)  # Move data to configured device
        # Move target to device and set correct dtype based on predict_mode
        if self.config.predict_mode == 'multi-class':
            target = target.to(self.device, dtype=torch.long)
        else:
            target = target.to(self.device, dtype=torch.float)
            demographic = demographic.to(self.device, dtype=torch.float)
        output = self.model(data).squeeze()  # Get model output
        loss = self.criterion(output, target)  # Calculate loss
        return data, target, output, loss  # Return batch data, target, output, and loss
    
    # Method to compute performance of model
    def performance(self):
    # Check if the task is classification (binary or multi-class) but not regression
    if self.config.label == 'sex' or (self.config.label == 'ahi' and self.config.predict_mode != 'regression'):
        accuracy, class_accuracy = self.compute_accuracy()  # Compute accuracy metrics
        # Print accuracy for each class
        for i in range(len(class_accuracy)):
            print('Accuracy of class {} : {:5.2f} %'.format(i, 100 * class_accuracy[i]))
        # Print overall accuracy
        print('Overall accuracy : {:5.2f} %'.format(accuracy * 100))
    else:
        # For regression tasks, calculate Mean Absolute Error (MAE) and Pearson correlation
        mae = np.mean(np.abs(self.targets-self.predictions))
        corr = pearsonr(self.targets, self.predictions)[0]
        print('Test MAE:', mae)
        print('Test Correlation:', corr)
    return mae, corr
    
    # Method to compute accuracy of model
    def compute_accuracy(self):
    # Calculate overall accuracy
    accuracy = (self.targets == self.predictions).sum() / len(self.targets)
    # Identify unique classes and their counts in the targets
    classes, class_counts = np.unique(self.targets, return_counts=True)
    # Initialize array to store correct predictions per class
    correct_class = np.zeros(len(classes))
    # Filter predictions that match the targets
    correct_categories = self.predictions[self.predictions == self.targets]
    # Count correct predictions per class
    correct_classes, correct_counts = np.unique(correct_categories, return_counts=True)
    for i in range(len(correct_classes)):
        correct_class[int(correct_classes[i])] = correct_counts[i]
    # Calculate accuracy per class
    class_accuracy = correct_class / class_counts
    return accuracy, class_accuracy

    # Method to visualize training and validation curves
    def visualize_train(self):
    # Different plots based on the task (classification vs. regression)
    if self.config.label == 'sex' or (self.config.label == 'ahi' and self.config.predict_mode != 'regression'):
        # Plot training and validation loss, and accuracy for classification tasks
        plt.plot(self.iterations, self.train_loss, self.iterations, self.val_loss, self.iterations, self.accuracy)
        plt.legend(('Train', 'Validation', 'Accuracy'))
        plt.ylabel('Cross Entropy Loss')
    else:
        # Plot training and validation loss for regression tasks
        plt.plot(self.iterations, self.train_loss, self.iterations, self.val_loss)
        plt.legend(('Train', 'Validation'))
        plt.ylabel('Mean Absolute Error')
    plt.xlabel('Iterations')
    plt.show()

        
        
    def visualize_predictions(self, predictions='largest', n_images=10):
        # Visualize specific predictions based on the argument ('largest', 'random', 'incorrect', 'correct')
        if predictions == 'largest':
            # For 'largest' errors, find indices of top errors and plot images
            for i in range(10):
                idx = self.subject_codes.index(self.df_errors['Subject Code'].iloc[i])
                plt.figure()
                plt.imshow(self.images[idx,:,:,:3])  # Assuming RGB images
                # Display target and prediction information on the image
                plt.title(self.subject_codes[idx])
                plt.text(170, 10, '{} = {:.1f}'.format(self.config.label, self.target_values[idx]))
                plt.text(180, 30, 'T = {:.1f}'.format(self.targets[idx]))
                plt.text(180, 50, 'P = {:.1f}'.format(self.predictions[idx]))
        else:
            # Handle 'random', 'incorrect', 'correct' predictions differently
            if predictions == 'random':
                samples = np.random.randint(self.images.shape[0]-1, size=n_images)
            elif predictions == 'incorrect':
                incorrect = np.where(self.targets != self.predictions)[0]
                samples = np.random.choice(incorrect, size=n_images, replace=False)
            elif predictions == 'correct':
                correct = np.where(self.targets == self.predictions)[0]
                samples = np.random.choice(correct, size=n_images, replace=False)
            for sample in samples:
                plt.figure()
                plt.imshow(self.images[sample,:,:,:3])
                plt.title(self.subject_codes[sample])
                plt.text(170, 10, '{} = {:.1f}'.format(self.config.label, self.target_values[sample]))
                plt.text(180,30,'T = {:.1f}'.format(self.targets[sample]))
                plt.text(180,50,'P = {:.1f}'.format(self.predictions[sample]))
                plt.show()
    
    # Method to generate a table with targets and predictions for each patient
    def errors_table(self):
        abs_diff = np.abs(self.targets-self.predictions)
        type_diff = np.sign(self.targets-self.predictions)
        test_summary = {'Subject Code': self.subject_codes, 'Targets': self.targets,
                        'Predictions': self.predictions, 'Differences': abs_diff,
                        'Type': type_diff}
        df = pd.DataFrame(test_summary)
        df['Type'] = df['Type'].replace({-1.0: 'Overestimate', 1.0: 'Underestimate'})
        df_errors = df.sort_values('Differences', ascending=False)        
            
        return df_errors
        
    # Method to save model checkpoint after an epoch    
    def save_checkpoint(self):
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.train_loss[-1],
            }, self.config.checkpoint_dir + 'checkpoint.tar')
        
    # Method to load model checkpoint for model evaluation
    def load_checkpoint(self):
        checkpoint = torch.load(self.config.checkpoint_dir + 'checkpoint.tar')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        self.test()

    # Main method to run training, validation, and testing loops
    def main(self):
        for epoch in range(self.config.epochs):  # Loop over the number of epochs
            # Adjust learning rate after 5 epochs
            if epoch == 5:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr / 10, weight_decay=self.config.weight_decay)
            self.train()  # Call train method
            self.validate()  # Call validate method
            self.test()  # Call test method
            mae, cc = self.performance()  # Call performance analysis method
        self.visualize_train()  # Visualize training and validation metrics
        self.visualize_predictions()  # Visualize predictions
