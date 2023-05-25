import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import os

from args import predict_parser
from model import Handler
from utils import get_data, SlidingWindowDataset, create_data_loader, pot_threshold, get_run_id, json_to_numpy

import mlflow

if __name__ == "__main__":

    # Get arguments from console
    parser = predict_parser()
    args = parser.parse_args()

    # Get custom id for every run
    id = datetime.now().strftime("%d%m%Y_%H%M%S")
    
    dataset = args.dataset

    experiment = mlflow.set_experiment(experiment_name=f"{dataset}_inference")
    exp_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=exp_id, run_name=id):

        # Get Production (or latest) model and run_id
        model_uri = f"models:/{dataset}_model/Production"
        try:
            model = mlflow.pytorch.load_model(model_uri)
            print(f"Fetched {dataset}_model from Production for predictions.")
            # get corresponding run_id
            client = mlflow.MlflowClient()
            run_id = client.get_latest_versions(f"{dataset}_model", stages=["Production"])[0].run_id
        except mlflow.exceptions.MlflowException:
            run_id = get_run_id("-1", f"{dataset}_training")
            model_uri = f"runs:/{run_id}/{dataset}_model"
            model = mlflow.pytorch.load_model(model_uri)
            print("No model found in Production stage, using model from latest run for predictions.")

        print(f"The model's run ID is: {run_id}")
        train_art_uri = mlflow.get_run(run_id).info.artifact_uri

        # Get configs used for model training
        model_parser = argparse.ArgumentParser()
        model_args, unknown = model_parser.parse_known_args()
        
        model_args.__dict__ = mlflow.artifacts.load_dict(train_art_uri+"/config.txt")

        window_size = model_args.window_size

        # --------------------------- START PREDICTION -----------------------------
        # Get data from the dataset
        (x_new, _) = get_data(dataset, mode="new", start=args.eval_start, end=args.eval_end)

        # This workaround need sto happen internally at the moment
        # We must use the last window_size timestamps from training as the first window_size timestamps
        # for evaluation, due to the sliding window framework
        x_train, _ = get_data(dataset, mode="train", start=-window_size, end=None)
        x_new = np.concatenate((x_train, x_new), axis=0)

        # Cast data into tensor objects
        x_new = torch.from_numpy(x_new).float()
        n_features = x_new.shape[1]

        # We want to perform forecasting/reconstruction on all features
        out_dim = n_features

        # Construct dataset from tensor objects - no stride here
        new_dataset = SlidingWindowDataset(x_new, window_size)

        print("Predicting:")
        # Create the data loader - no shuffling here
        new_loader, _ = create_data_loader(new_dataset, model_args.batch_size, None, False)

        # Initialize the Handler module
        handler = Handler(
            model=model,
            optimizer=None,
            scheduler=None,
            window_size=window_size,
            n_features=n_features,
            batch_size=model_args.batch_size,
            n_epochs=None,
            patience=None,
            forecast_criterion=None,
            recon_criterion=None,
            use_cuda=args.use_cuda,
            print_every=None,
            gamma=model_args.gamma
        )

        # Get new scores (inference needs to be fast, no details needed)
        print("Calculating scores on new data...")
        new_scores = handler.score(loader=new_loader, details=False)

        # Calculate threshold via POT based on the new_scores
        if args.threshold == "POT":
            # Load stored scores for training data
            print("Loading scores from data used for training...")
            train_scores = json_to_numpy(train_art_uri+"/anom_scores.json")
            
            if args.use_mov_av:
                smoothing_window = int(model_args.batch_size * window_size * 0.05)
                train_scores = pd.DataFrame(train_scores).ewm(span=smoothing_window).mean().values.flatten()
                new_scores = pd.DataFrame(new_scores).ewm(span=smoothing_window).mean().values.flatten()

            pot_thresh = pot_threshold(train_scores, new_scores, q=args.q, level=args.level, dynamic=args.dynamic_pot)

            # Log the POT threshold as part of this run, do not override anything from training/eval
            mlflow.log_dict({"POT": pot_thresh}, "thresholds.json")

            threshold = pot_thresh
        # Pick among the selected thresholds
        else:
            thresholds = mlflow.artifacts.load_dict(train_art_uri+"/thresholds.json")
            threshold = thresholds["epsilon"]

        print(f"Predicting anomalies based on {args.threshold}-generated threshold - threshold value: {threshold}")

        # Make predictions based on threshold
        anomalies = handler.predict(new_scores, threshold)
        
        # ---------------------------- END PREDICTION ------------------------------

        # save results
        with open('anomalies.txt', 'w') as f:
            for anom in anomalies:
                f.write(f"{anom}\n")
        mlflow.log_artifact('anomalies.txt')
        os.remove('anomalies.txt')

    print("Finished.")