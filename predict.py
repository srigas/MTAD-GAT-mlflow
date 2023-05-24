import argparse
import json
import numpy as np
import pandas as pd
import torch
import os

from args import predict_parser
from architecture import MTAD_GAT
from model import Handler
from utils import get_model_id, get_data, SlidingWindowDataset, create_data_loader, pot_threshold, update_json

if __name__ == "__main__":

    # Get arguments from console
    parser = predict_parser()
    args = parser.parse_args()

    dataset = args.dataset

    model_id = get_model_id(args.model_id, dataset)

    model_path = f"./output/{dataset}/{model_id}"

    # Check that the model exists
    if not os.path.isfile(f"{model_path}/model.pt"):
        raise Exception(f"<{model_path}/model.pt> does not exist.")

    # Get configs used for model training
    print(f'Using model from {model_path}')
    model_parser = argparse.ArgumentParser()
    model_args, unknown = model_parser.parse_known_args()
    model_args_path = f"{model_path}/config.txt"

    with open(model_args_path, "r") as f:
        model_args.__dict__ = json.load(f)
    window_size = model_args.window_size

    # Check that model is trained on the specified dataset
    if args.dataset.lower() != model_args.dataset.lower():
        raise Exception(f"Model trained on {model_args.dataset}, but asked to predict {args.dataset}.")

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
    # TODO: Implement horizon parameter
    new_dataset = SlidingWindowDataset(x_new, window_size)

    print("Predicting:")
    # Create the data loader - no shuffling here
    new_loader, _ = create_data_loader(new_dataset, model_args.batch_size, None, False)

    # Initialize the model
    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        kernel_size=model_args.kernel_size,
        use_gatv2=model_args.use_gatv2,
        feat_gat_embed_dim=model_args.feat_gat_embed_dim,
        time_gat_embed_dim=model_args.time_gat_embed_dim,
        gru_n_layers=model_args.gru_n_layers,
        gru_hid_dim=model_args.gru_hid_dim,
        forecast_n_layers=model_args.fc_n_layers,
        forecast_hid_dim=model_args.fc_hid_dim,
        recon_n_layers=model_args.recon_n_layers,
        recon_hid_dim=model_args.recon_hid_dim,
        dropout=model_args.dropout,
        alpha=model_args.alpha
    )

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
        dload=None,
        print_every=None,
        gamma=model_args.gamma
    )

    # Load the trained model with the given model_id
    handler.load(f"{model_path}/model.pt")

    # Get new scores (inference needs to be fast, no details needed)
    print("Calculating scores on new data...")
    new_scores = handler.score(loader=new_loader, details=False)

    # Calculate threshold via POT based on the new_scores
    if args.threshold == "POT":
        # Load stored scores for training data
        print("Loading scores from data used for training...")
        train_scores = np.load(f'{model_path}/anom_scores.npy')

        if args.use_mov_av:
            smoothing_window = int(model_args.batch_size * window_size * 0.05)
            train_scores = pd.DataFrame(train_scores).ewm(span=smoothing_window).mean().values.flatten()
            new_scores = pd.DataFrame(new_scores).ewm(span=smoothing_window).mean().values.flatten()

        pot_thresh = pot_threshold(train_scores, new_scores, q=args.q, level=args.level, dynamic=args.dynamic_pot)

        update_json(f'{model_path}/thresholds.json', {"POT": pot_thresh})

        threshold = pot_thresh
    # Pick among the selected thresholds
    else:
        with open(f'{model_path}/thresholds.json') as json_file:
            thresholds = json.load(json_file)
        threshold = thresholds[args.threshold]

    print(f"Predicting anomalies based on {args.threshold}-generated threshold - threshold value: {threshold}")

    # Make predictions based on threshold
    anomalies = handler.predict(new_scores, threshold)
    
    # ---------------------------- END PREDICTION ------------------------------

    # save results
    with open(f'{model_path}/anomalies.txt', 'w') as f:
        for anom in anomalies:
            f.write(f"{anom}\n")

    print("Finished.")