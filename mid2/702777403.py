"""
This template will help you solve the coding problem in the midterm.

This script contains a function to help you read the command-line 
arguments and some other to give you an initial structure with some
hints in the comments.

Import the necessary libraries and define any functions required for 
your implementation.

"""

import argparse


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

# Create your model
class Model(torch.nn.Module):
    def __init__(self, input_size=300, num_classes=10):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(256, 128)

        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.leaky_relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.leaky_relu(x)

        x = self.fc3(x)
        x = nn.functional.leaky_relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = nn.functional.leaky_relu(x)

        x = self.output(x)
        return x

# Create your data loader
def data_loader(data, labels, batch_size, shuffle=True):
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32),
                            torch.tensor(labels, dtype=torch.uint8))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def read_args():
    """This function reads the arguments 
    from the command line. 

    Returns:
        list with the input arguments
    """
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("true"):
            return True
        elif v.lower() in ("false"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()

    parser.add_argument("is_testing", type=str2bool, help="Set Testing Stage")
    parser.add_argument("ckpt_path", nargs='?', type=str, help="Checkpoint path", default='./')
    parser.add_argument("testing_data_path", nargs='?', type=str, help="Testing data path", default='./')
    parser.add_argument("output_path", nargs='?', type=str, help="output path", default='./')

    return parser.parse_args()

def train():
    """
    Do everything related to the training of your model
    """
    # Define hyperparameters and constants
    batch_size = 32
    k = 10
    num_epochs = 50
    early_stopping_patience = 8

    # Initialize lists to track metrics across folds
    fold_train_losses = []
    fold_train_accuracies = []
    fold_val_losses = []
    fold_val_accuracies = []

    # Load your data and labels
    data = np.load("midterm2_training_data.npy")
    labels = np.load("midterm2_training_labels.npy")

    # Initialize KFold cross-validator
    kf = KFold(n_splits=k, shuffle=True, random_state=studentname_udid)

    # Variables to track the best model across all folds
    best_overall_val_loss = np.inf
    best_overall_model_state = None

    # Start k-fold cross-validation
    for fold, (train_index, val_index) in enumerate(kf.split(data)):
        logging.info(f"Starting fold {fold + 1}")

        # Split data for the current fold
        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        # Create datasets and dataloaders
        train_loader = data_loader(X_train, y_train, batch_size)

        val_loader = data_loader(X_val, y_val, batch_size, shuffle=False)

        logging.info(f"Data loaded for fold {fold + 1}. Starting training...")

        # Instantiate the model
        model = Model()

        # Setup optimizer, loss criterion
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)

        # Variables to track early stopping
        best_val_loss = np.inf
        epochs_without_improvement = 0

        # Lists to track metrics for the current fold
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        # Training loop for the current fold
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            # Iterate over the training data
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == y_batch).sum().item()
                total_samples += y_batch.size(0)

            train_loss = running_loss / total_samples
            train_accuracy = correct_predictions / total_samples

            # Append training metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == y_batch).sum().item()
                    total_samples += y_batch.size(0)

            val_loss /= total_samples
            val_accuracy = correct_predictions / total_samples

            # Append validation metrics
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Print training and validation statistics
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save the best model state for this fold
                best_model_state = model.state_dict()
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Save the model if it has the best validation loss across all folds
        if best_val_loss < best_overall_val_loss:
            best_overall_val_loss = best_val_loss
            best_overall_model_state = best_model_state

        # Append fold metrics to the overall lists
        fold_train_losses.append(train_losses)
        fold_train_accuracies.append(train_accuracies)
        fold_val_losses.append(val_losses)
        fold_val_accuracies.append(val_accuracies)

    # Save the best model across all folds
    torch.save(best_overall_model_state, f'{studentname_udid}.pth')

    print('Finish training')

def test(input_args):
    """
    Do everything related to the testing of your model
    """

    print('Start testing')
    # Read data from input_args.testing_data_path
    data = np.load(input_args.testing_data_path)

    # Load model with weights from input_args.ckpt_path
    model = Model()
    model.load_state_dict(torch.load(input_args.ckpt_path))

    logging.info("Model loaded. Starting predictions...")

    # Compute model predictions
    model.eval()
    # Compute model predictions
    with torch.no_grad():
        inputs = torch.tensor(data, dtype=torch.float32)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

    logging.info("Predictions done. Saving results...")

    # Save predictions as [studentname_udid].npy file in input_args.output_path
    np.save(f"{input_args.output_path}/{studentname_udid}.npy", predictions.numpy())


def main():
    """This is the main function of yor script.
    From here you call your training or 
    testing functions.
    """
    input_args = read_args()

    if input_args.is_testing:
        test(input_args)
    else:
        train()

if __name__=="__main__":
    # Define global variables
    studentname_udid = 702777403
    torch.manual_seed(42) 
    np.random.seed(studentname_udid)
    main()
