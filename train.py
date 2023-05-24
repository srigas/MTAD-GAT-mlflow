from datetime import datetime
import torch.nn as nn
import torch
import pandas as pd
import os
import numpy as np
import json

from args import train_parser
from architecture import MTAD_GAT
from model import Handler
from utils import get_data, SlidingWindowDataset, create_data_loader, find_epsilon, update_json


if __name__ == "__main__":

    # Get arguments from console
    parser = train_parser()
    args = parser.parse_args()

    dataset = args.dataset
    window_size = args.window_size

    # Set output path for specific dataset
    output_path = f'output/{dataset}'

    # Get custom id for every run
    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    # Setup an additional directory where the results of each different model
    # for the same dataset are stored
    save_path = f"{output_path}/{id}"

    # Make directories if they don't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # --------------------------- START TRAINING -----------------------------
    # Get data from the dataset
    (x_train, _) = get_data(dataset, mode="train", start=args.train_start, end=args.train_end)

    # Cast data into tensor objects
    x_train = torch.from_numpy(x_train).float()
    n_features = x_train.shape[1]

    # We want to perform forecasting/reconstruction on all features
    out_dim = n_features
    print(f"Proceeding with forecasting and reconstruction of all {n_features} input features.")

    # Construct dataset from tensor object
    train_dataset = SlidingWindowDataset(x_train, window_size, args.stride)

    print("Training:")
    # Create the data loader(s)
    train_loader, val_loader = create_data_loader(train_dataset, args.batch_size, 
                                                  args.val_split, args.shuffle_dataset)

    # Initialize the model
    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        kernel_size=args.kernel_size,
        use_gatv2=args.use_gatv2,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha
    )

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)

    # Add a scheduler for variable learning rate
    e_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_lr, gamma=args.gamma_lr)

    # Set the criterion for each process: forecasting & reconstruction
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    # Initialize the Handler module
    handler = Handler(
        model=model,
        optimizer=optimizer,
        scheduler=e_scheduler,
        window_size=window_size,
        n_features=n_features,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        patience=args.patience,
        forecast_criterion=forecast_criterion,
        recon_criterion=recon_criterion,
        use_cuda=args.use_cuda,
        dload=save_path,
        print_every=args.print_every,
        gamma=args.gamma
    )

    # Start training
    handler.fit(train_loader, val_loader)

    # ---------------------------- END TRAINING ------------------------------

    # Get scores for training data to be used for thresholds later on
    print("Calculating scores on training data to be used for thresholding...")
    anom_scores = handler.score(loader=train_loader, details=False)
    # Also get the ones from the validation data
    if val_loader is not None:
        val_scores = handler.score(loader=val_loader, details=False)
        anom_scores = np.concatenate((anom_scores, val_scores), axis=0)

    # get threshold using epsilon method
    if str(args.reg_level).lower() != "none":

        if args.use_mov_av:
            smoothing_window = int(args.batch_size * window_size * 0.05)
            anom_scores = pd.DataFrame(anom_scores).ewm(span=smoothing_window).mean().values.flatten()

        e_thresh = find_epsilon(errors=anom_scores, reg_level=args.reg_level)
        # save the value to be used later on
        update_json(f'{save_path}/thresholds.json', {"epsilon": e_thresh})

    # save scores to be used later on for other thresholds
    with open(f'{save_path}/anom_scores.npy', 'wb') as f:
        np.save(f, anom_scores)

    # Save the losses in .npy files to be used (e.g. for plotting)
    loss_path = os.path.join(save_path, "losses")
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)

    for key in handler.losses:
        with open(f'{loss_path}/{key}.npy', 'wb') as f:
            np.save(f, handler.losses[key])

    # Workaround to write dimensions of dataset in config
    # to be used with the Plotter method
    args.__dict__['n_features'] = out_dim
    
    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)

    print("Finished.")
