# Import necessary libraries for the class functionality
import torch
import numpy as np
import random
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from python_json_config import ConfigBuilder
from psg_dataset import PSGDataset
from models import TransferNetwork
from transformations import Rescale, Normalize, ToTensor

# Define the main class for training models
class CrossValidationTrainer:
    def __init__(self, config_path):
        # Initialize the class with the path to a configuration file
        self.builder = ConfigBuilder()  # Instantiate a ConfigBuilder for parsing the config file
        self.config = self.builder.parse_config(config_path)  # Parse the configuration file
        # Adjust the number of channels based on the configuration, supporting even or odd numbers of images
        self.config.n_channels = 2 * self.config.n_images if (self.config.n_images % 2) == 0 else 3 * self.config.n_images
        # Determine the device (CPU or CUDA) based on CUDA availability for training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize the dataset with transformations applied to the data
        self.dataset = PSGDataset(config=self.config, transform=transforms.Compose([Rescale(224), Normalize(), ToTensor()]))
        # Prepare the data for training by combining samples and shuffling
        self.prepare_data()

    def prepare_data(self):
        # Combine training and test samples, then shuffle the combined dataset
        self.dataset = self.dataset.train_samples + self.dataset.test_samples
        random.seed(42)  # Seed for reproducibility
        random.shuffle(self.dataset)
        # Split the dataset into folds for cross-validation
        n_folds = 10
        fold_samples = round(len(self.dataset) / n_folds)
        self.folds = [self.dataset[i:i + fold_samples] for i in range(0, len(self.dataset), fold_samples)]

    def train_model(self, n_epochs=10, batch_size=8):
        # Initialize lists to store results from each fold
        results = []
        for i in range(len(self.folds)):
            # Get the current folds for training, validation, and testing
            test_fold, val_fold, train_folds = self.get_folds(i)
            # Loaders for batch processing of each set
            train_loader = DataLoader(train_folds, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_fold, batch_size=batch_size, drop_last=True)
            test_loader = DataLoader(test_fold, batch_size=10)

            # Initialize the model, optimizer, and loss function for the current fold
            model, optimizer, criterion = self.initialize_model()
            print(f'Training fold {i+1}')

            # Train the model for the specified number of epochs
            self.train_epochs(model, optimizer, criterion, train_loader, val_loader, n_epochs, i)
            # Evaluate the model on the test set and store the results
            fold_results = self.evaluate_model(model, test_loader)
            results.append(fold_results)

        # Save the results from all folds to an Excel file
        self.saveresults(results)

    def get_folds(self, i):
        # Extract test and validation folds, and combine the remaining folds for training
        test_fold = self.folds.pop(0)
        val_fold = self.folds.pop(0)
        train_folds = [item for sublist in self.folds for item in sublist]
        return test_fold, val_fold, train_folds

    def initialize_model(self):
        # Instantiate the model, optimizer, and loss criterion based on the configuration
        model = TransferNetwork(config=self.config)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        criterion = model.loss
        model.to(self.device)  # Move the model to the appropriate device
        return model, optimizer, criterion

    def train_epochs(self, model, optimizer, criterion, train_loader, val_loader, n_epochs, fold_index):
        # Train the model for a specified number of epochs, performing backpropagation and optimization
        for epoch in range(n_epochs):
            running_loss = 0.0
            model.train()  # Set the model to training mode
            for _, batch in enumerate(train_loader, 0):
                loss = self._process_batch(batch, model, criterion, optimizer)
                running_loss += loss.item()
            mean_loss = running_loss / len(train_loader)
            print(f'Epoch: {epoch + 1}, Train loss: {mean_loss}')

            # Validate the model at the end of each epoch to monitor performance and potentially save the best model
            val_mean_loss = self.validate_model(model, val_loader, criterion)
            self.save_checkpoint(model, optimizer, mean_loss, val_mean_loss, fold_index)

    def process_batch(self, batch, model, criterion, optimizer):
        # Process a single batch of data: forward pass, calculate loss, backward pass (if training), and update model parameters
        data, target, demographics = batch['image'], batch['target'], batch['demographic']
        data, target, demographics = data.to(self.device, dtype=torch.float), target.to(self.device), demographics.to(self.device, dtype=torch.float)
        if optimizer:  # Check if in training mode
            optimizer.zero_grad()
        output = model(data, demographics).squeeze()
        loss = criterion(output, target)
        if optimizer:  # Only perform backpropagation and optimization if in training mode
            loss.backward()
            optimizer.step()
        return loss

    def validate_model(self, model, val_loader, criterion):
        # Validate the model on the validation set, calculating the average loss over all batches
        val_running_loss = 0.0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            for _, val_batch in enumerate(val_loader, 0):
                loss = self.process_batch(val_batch, model, criterion, None)  # No optimizer for validation
                val_running_loss += loss.item()
        return val_running_loss / len(val_loader)

    def save_checkpoint(self, model, optimizer, train_loss, val_mean_loss, fold_index):
        # Save a model checkpoint if it has the best validation performance so far
        torch.save({
            'epoch': len(train_loss),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, self.config.checkpoint_dir + f'checkpoint{fold_index}.tar')

    def evaluate_model(self, model, test_loader):
        # Evaluate the model on the test set, collecting targets, predictions, and subject codes
        targets, predictions, subjects = [], [], []
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            for _, test_batch in enumerate(test_loader, 0):
                data_test, target_test, demographics_test = test_batch['image'], test_batch['target'], test_batch['demographic']
                data_test, target_test, demographics_test = data_test.to(self.device, dtype=torch.float), target_test.to(self.device), demographics_test.to(self.device, dtype=torch.float)
                output_test = model(data_test, demographics_test).squeeze()
                targets.extend(target_test.cpu().numpy())
                predictions.extend(output_test.cpu().numpy())
                subjects.extend(test_batch['subject_code'])
        return {'subjects': subjects, 'targets': targets, 'predictions': predictions}

    def save_results(self, results):
        # Compile and save the results from all folds to an Excel file
        all_subjects, all_targets, all_predictions = [], [], []
        for result in results:
            all_subjects.extend(result['subjects'])
            all_targets.extend(result['targets'])
            all_predictions.extend(result['predictions'])
        summary = {'Subjects': all_subjects, 'Targets': all_targets, 'Predictions': all_predictions}
        df = pd.DataFrame(summary)
        df.to_excel(self.config.checkpoint_dir + '/Predictions.xlsx')

# Example of how to use the class
if __name__ == "__main__":
    config_path = 'C:/Users/umaer/OneDrive/Documents/PhD/Code/predictDemographics/config.json'
    trainer = CrossValidationTrainer(config_path)
    trainer.train_model()
