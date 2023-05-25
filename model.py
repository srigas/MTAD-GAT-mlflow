import os
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

import mlflow


class Handler:
    """Handler class for the model defined in architecture.py

    :param model: model
    :param optimizer: Optimizer used to minimize the loss function
    :param scheduler: Scheduler used to vary learning rate
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param batch_size: size of batches for loaders
    :param n_epochs: Number of iterations/epochs
    :param patience: Number of steps to wait before activating Early Stopping
    :param forecast_criterion: Loss to be used for forecasting.
    :param recon_criterion: Loss to be used for reconstruction.
    :param use_cuda: To be run on GPU or not (boolean)
    :param print_every: At what epoch interval to print losses
    :param gamma: Gamma parameter for getting anomaly scores
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        window_size,
        n_features,
        batch_size=256,
        n_epochs=200,
        patience=None,
        forecast_criterion=nn.MSELoss(),
        recon_criterion=nn.MSELoss(),
        use_cuda=True,
        print_every=1,
        gamma=1.0
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.window_size = window_size
        self.n_features = n_features
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.forecast_criterion = forecast_criterion
        self.recon_criterion = recon_criterion
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.print_every = print_every
        self.gamma = gamma

        self.epoch_times = []

        if self.device == "cuda":
            self.model.cuda()

    def pass_epoch(self, loader, mode):
        """Pass through a data loader for an epoch and either do backprop for training
        or simply calculate losses for validation
        :param loader: loader of input data
        :param mode: "train" in order to do backpropagation for learning
        """
        if mode=="train":
            self.model.train() # Set to training mode
        else:
            self.model.eval() # Set to eval mode
        
        forecast_losses = []
        recon_losses = []

        grad_mode = True if mode=="train" else False

        with torch.set_grad_enabled(grad_mode):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                if mode=="train":
                    self.optimizer.zero_grad()

                y_hat, _ = self.model(x)

                # Shifting input to include the observed value (y) when doing the reconstruction
                # i.e. remove the first timestamp from the window and append the observed value
                recon_x = torch.cat((x[:, 1:, :], y), dim=1)
                _, window_recon = self.model(recon_x)

                # In case objects are in shape (batch_size, 1, n_features)
                if y_hat.ndim == 3:
                    y_hat = y_hat.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)

                # Calculates RMSE given the MSE Loss functions
                forecast_loss = torch.sqrt(self.forecast_criterion(y, y_hat))
                recon_loss = torch.sqrt(self.recon_criterion(x, window_recon))

                if mode=="train":
                    loss = forecast_loss + recon_loss
                    loss.backward()
                    self.optimizer.step()

                forecast_losses.append(forecast_loss.item())
                recon_losses.append(recon_loss.item())

        forecast_losses = np.array(forecast_losses)
        recon_losses = np.array(recon_losses)

        forecast_epoch_loss = np.sqrt((forecast_losses ** 2).mean())
        recon_epoch_loss = np.sqrt((recon_losses ** 2).mean())

        return forecast_epoch_loss, recon_epoch_loss


    def fit(self, train_loader, val_loader=None):
        """Train model for self.n_epochs.
        Train and validation (if validation loader given) losses stored in self.losses

        :param train_loader: train loader of input data
        :param val_loader: validation loader of input data
        """
        # Initialize these for early stopping
        min_val_loss, stopping_ct = 9999, 0

        print(f"Training model for {self.n_epochs} epoch(s)...")
        train_start = time.time() # start timing all fitting process
        for epoch in range(self.n_epochs):
            print(f"[Epoch {epoch + 1}]")
            epoch_start = time.time() # start timing epoch
            
            train_fc_loss, train_rc_loss = self.pass_epoch(train_loader, "train")
            total_train_loss = train_fc_loss+train_rc_loss
            # Vary learning rate
            self.scheduler.step()

            mlflow.log_metric(key="train_fc_loss", value=train_fc_loss, step=epoch+1)
            mlflow.log_metric(key="train_rc_loss", value=train_rc_loss, step=epoch+1)
            mlflow.log_metric(key="total_train_loss", value=total_train_loss, step=epoch+1)

            # Evaluate on validation set if a val_loader is provided
            if val_loader is not None:
                val_fc_loss, val_rc_loss = self.pass_epoch(val_loader, None)
                total_val_loss = val_fc_loss+val_rc_loss

                mlflow.log_metric(key="val_fc_loss", value=val_fc_loss, step=epoch+1)
                mlflow.log_metric(key="val_rc_loss", value=val_rc_loss, step=epoch+1)
                mlflow.log_metric(key="total_val_loss", value=total_val_loss, step=epoch+1)

                if self.patience is not None:
                    # Save the model only if val loss decreased
                    if min_val_loss > total_val_loss:
                        min_val_loss = total_val_loss
                        self.save("model.pt")
                        stopping_ct = 0
                    # Otherwise increase the stopping counter by one
                    else:
                        print(f'Validation Loss Increased. Early Stopping counter [{stopping_ct+1}/{self.patience}].')
                        stopping_ct += 1

                    if stopping_ct == self.patience:
                        print('Early Stopping counter reached patience limit.')
                        print('Terminating training and reverting back to last stable version.')
                        break
                else:
                    self.save("model.pt")

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            if epoch % self.print_every == 0:
                s = (
                    f"Elapsed time: {epoch_time:.1f}s\n"
                    f"Forecasting Loss: {train_fc_loss:.5f},\t"
                    f"Reconstruction Loss: {train_rc_loss:.5f},\t"
                    f"Total Training Loss: {total_train_loss:.5f}."
                )

                if val_loader is not None:
                    s += (
                        "\n"
                        f"Forecasting Loss: {val_fc_loss:.5f},\t"
                        f"Reconstruction Loss: {val_rc_loss:.5f},\t"
                        f"Total Validation Loss: {total_val_loss:.5f}."
                    )

                print(s)
                
        if val_loader is not None:
            self.load("model.pt")
            os.remove("model.pt") # it is logged through mlflow either way

        train_time = int(time.time() - train_start)
        print(f"\nTraining finished after {train_time}s.")

    def score(self, loader, details=False):
        """Method that calculates anomaly scores
        :param loader: loader of input data
        :param details: bool to specify if additional info is to be returned
        :return np array of anomaly scores + dataframe of details
        """
        self.model.eval()
        preds, recons, actual = [], [], []
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat, _ = self.model(x)

                # Shifting input to include the observed value (y) when doing the reconstruction
                recon_x = torch.cat((x[:, 1:, :], y), dim=1)
                _, window_recon = self.model(recon_x)

                # In case objects are in shape (batch_size, 1, n_features)
                if y_hat.ndim == 3:
                    y_hat = y_hat.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)

                preds.append(y_hat.detach().cpu().numpy())
                actual.append(y.detach().cpu().numpy().squeeze())
                # Extract last reconstruction only
                recons.append(window_recon[:, -1, :].detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        recons = np.concatenate(recons, axis=0)
        actual = np.concatenate(actual, axis=0)

        anomaly_scores = (np.sqrt((preds[:,:]-actual[:,:])**2) + self.gamma*np.sqrt(
                (recons[:,:]-actual[:,:])**2))/(1.0+self.gamma)
        
        anomaly_scores = np.sum(anomaly_scores, 1)

        if details:
            df = {}
            for i in range(self.n_features):
                df[f"FC_{i}"] = preds[:,i]
                df[f"RECON_{i}"] = recons[:,i]
                df[f"TRUE_{i}"] = actual[:,i]

                df[f"SCORE_{i}"] = (np.sqrt((preds[:,i]-actual[:,i])**2) + self.gamma*np.sqrt(
                    (recons[:,i]-actual[:,i])**2))/(1.0+self.gamma)
                
            df = pd.DataFrame(df)
            return anomaly_scores, df

        return anomaly_scores
    
    def predict(self, scores, threshold):
        """Method that predicts anomalies given scores and a threshold
        :param scores: np array of anomaly scores
        :param threshold: threshold that separates anomalous from non anomalous values
        :return list of 0s and 1s corresponding to indices of scores
        """
        anomalies = [0 if score < threshold else 1 for score in scores]

        return anomalies

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as
        """
        torch.save(self.model.state_dict(), file_name)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        """
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))