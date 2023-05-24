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
from utils import get_metrics, PA, calculate_latency

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
        raise Exception(f"Model trained on {model_args.dataset}, but asked to evaluate {args.dataset}.")

    # --------------------------- START EVALUATION -----------------------------
    # Get data from the dataset
    (x_eval, y_eval) = get_data(dataset, mode="eval", start=args.eval_start, end=args.eval_end)

    # This workaround need sto happen internally at the moment
    # We must use the last window_size timestamps from training as the first window_size timestamps
    # for evaluation, due to the sliding window framework
    x_train, _ = get_data(dataset, mode="train", start=-window_size, end=None)
    x_eval = np.concatenate((x_train, x_eval), axis=0)

    # Cast data into tensor objects
    x_eval = torch.from_numpy(x_eval).float()
    n_features = x_eval.shape[1]

    # We want to perform forecasting/reconstruction on all features
    out_dim = n_features

    # Construct dataset from tensor objects - no stride here
    # TODO: Implement horizon parameter
    eval_dataset = SlidingWindowDataset(x_eval, window_size)

    print("Evaluating:")
    # Create the data loader - no shuffling here
    eval_loader, _ = create_data_loader(eval_dataset, model_args.batch_size, None, False)

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

    # Get new scores
    print("Calculating scores on evaluation data...")
    new_scores, df = handler.score(loader=eval_loader, details=args.show_details)

    # Save a dataframe with per-feature predictions and scores for debugging
    if args.show_details:
        df.to_pickle(f'{model_path}/per_feat_scores.pkl')
        with open(f'{model_path}/eval_scores.npy', 'wb') as f:
            np.save(f, new_scores)

    # --> POT threshold
    
    print("Loading scores from data used for training...")
    train_scores = np.load(f'{model_path}/anom_scores.npy')
    
    print(train_scores.shape)

    if args.use_mov_av:
        smoothing_window = int(model_args.batch_size * window_size * 0.05)
        train_scores = pd.DataFrame(train_scores).ewm(span=smoothing_window).mean().values.flatten()
        new_scores = pd.DataFrame(new_scores).ewm(span=smoothing_window).mean().values.flatten()

    pot_thresh = pot_threshold(train_scores, new_scores, q=args.q, level=args.level, dynamic=args.dynamic_pot)

    print(f"Predicting anomalies based on POT-generated threshold - threshold value: {pot_thresh:.4f}")

    # Make predictions based on threshold
    pot_anoms = handler.predict(new_scores, pot_thresh)

    # Perform evaluation based on predictions
    f1_pot, prec_pot, rec_pot = get_metrics(pot_anoms, y_eval)
    
    # Get delays
    pot_correct, pot_delay, pot_identified, pot_unidentified = calculate_latency(y_eval, pot_anoms)

    print(f"Without point-adjustment, the F1-Score is: {f1_pot:.4f}")

    # Same evaluation, but with point adjustment
    pa_pot_anoms = PA(y_eval, pot_anoms)
    pa_f1_pot, pa_prec_pot, pa_rec_pot = get_metrics(pa_pot_anoms, y_eval)

    print(f"With point-adjustment, the F1-Score is: {pa_f1_pot:.4f}\n")
    print(f"A total of {pot_correct} events were correctly identified. Here are the delays for each anomaly range:")
    print(pot_identified)
    print(f"The average latency is {pot_delay:.4f} timestamps.")
    print("The following events were not identified:")
    print(pot_unidentified)

    update_json(f'{model_path}/thresholds.json', {"POT": pot_thresh})

    # --> epsilon threshold

    with open(f'{model_path}/thresholds.json') as json_file:
        thresholds = json.load(json_file)
    e_thresh = float(thresholds["epsilon"])

    print(f"Predicting anomalies based on epsilon-generated threshold - threshold value: {e_thresh:.4f}")

    # Make predictions based on threshold
    e_anoms = handler.predict(new_scores, e_thresh)

    # Perform evaluation based on predictions
    f1_e, prec_e, rec_e = get_metrics(e_anoms, y_eval)
    
    # Get delays
    e_correct, e_delay, e_identified, e_unidentified = calculate_latency(y_eval, e_anoms)

    print(f"Without point-adjustment, the F1-Score is: {f1_e:.4f}")

    # Same evaluation, but with point adjustment
    pa_e_anoms = PA(y_eval, e_anoms)
    pa_f1_e, pa_rec_e, pa_prec_e = get_metrics(pa_e_anoms, y_eval)

    print(f"With point-adjustment, the F1-Score is: {pa_f1_e:.4f}\n")
    print(f"A total of {e_correct} events were correctly identified. Here are the delays for each anomaly range:")
    print(e_identified)
    print(f"The average latency is {e_delay:.4f} timestamps.")
    print("The following events were not identified:")
    print(e_unidentified)

    # --> Brute-force threshold (Best F1)

    print("Initiating Brute-force method for best F1...")

    thresholds = np.linspace(min(new_scores), max(new_scores), 200)
    bf_res, pa_bf_res = [], []
    for bf_thresh in thresholds:
        anoms = handler.predict(new_scores, bf_thresh)
        # Get F1-Score without PA
        f1, prec, rec = get_metrics(anoms, y_eval)
        bf_res.append(f1)
        # Get F1-Score with PA
        pa_anoms = PA(y_eval, anoms)
        f1, prec, rec = get_metrics(pa_anoms, y_eval)
        pa_bf_res.append(f1)
    
    best_idx = bf_res.index(max(bf_res))
    bf_thresh = thresholds[best_idx]

    # Run one last time
    bf_anoms = handler.predict(new_scores, bf_thresh)
    f1_bf, prec_bf, rec_bf = get_metrics(bf_anoms, y_eval)
    
    # Get delays
    bf_correct, bf_delay, bf_identified, bf_unidentified = calculate_latency(y_eval, bf_anoms)
    
    print(f"Without point-adjustment, the best achievable F1-Score is: {f1_bf:.4f}")
    print(f"This is achieved by setting the threshold at: {bf_thresh:.4f}\n")
    
    pa_bf_anoms = PA(y_eval, bf_anoms)
    pa_f1_bf, pa_rec_bf, pa_prec_bf = get_metrics(pa_bf_anoms, y_eval)

    print(f"The corresponding point-adjusted F1-Score is: {pa_f1_bf:.4f}\n")
    print(f"A total of {bf_correct} events were correctly identified. Here are the delays for each anomaly range:")
    print(bf_identified)
    print(f"The average latency is {bf_delay:.4f} timestamps.")
    print("The following events were not identified:")
    print(bf_unidentified)

    update_json(f'{model_path}/thresholds.json', {"BF-F1": bf_thresh})
    
    # --> Brute-force threshold (Best adjusted F1)

    print("Initiating Brute-force method for best point-adjusted F1...")
    
    pa_best_idx = pa_bf_res.index(max(pa_bf_res))
    pa_bf_thresh = thresholds[pa_best_idx]
    
    # Run one last time
    bf_anoms_2 = handler.predict(new_scores, pa_bf_thresh)
    f1_bf_2, prec_bf_2, rec_bf_2 = get_metrics(bf_anoms_2, y_eval)
    
    # Get delays
    bf_correct_2, bf_delay_2, bf_identified_2, bf_unidentified_2 = calculate_latency(y_eval, bf_anoms_2)
    
    print(f"Without point-adjustment, the best achievable F1-Score is: {f1_bf_2:.4f}")
    print(f"This is achieved by setting the threshold at: {pa_bf_thresh:.4f}\n")
    
    pa_bf_anoms_2 = PA(y_eval, bf_anoms_2)
    pa_f1_bf_2, pa_rec_bf_2, pa_prec_bf_2 = get_metrics(pa_bf_anoms_2, y_eval)

    print(f"The corresponding point-adjusted F1-Score is: {pa_f1_bf_2:.4f}\n")
    print(f"A total of {bf_correct_2} events were correctly identified. Here are the delays for each anomaly range:")
    print(bf_identified_2)
    print(f"The average latency is {bf_delay_2:.4f} timestamps.")
    print("The following events were not identified:")
    print(bf_unidentified_2)

    update_json(f'{model_path}/thresholds.json', {"BF-F1-PA": pa_bf_thresh})

    # ---------------------------- END EVALUATION ------------------------------

    # save results
    with open(f'{model_path}/eval_summary.txt', 'w') as f:

        lines_to_write = [
            "POT algorithm:\n",
            f"Threshold: {pot_thresh:.4f}\n",
            f"Without PA:\t F1-Score: {f1_pot*100:.4f}%, \t Recall: {rec_pot*100:.4f}%, \t Precision: {prec_pot*100:.4f}%\n",
            f"With PA:\t {pa_f1_pot*100:.4f}%, \t Recall: {pa_rec_pot*100:.4f}%, \t Precision: {pa_prec_pot*100:.4f}%\n",
            f"Identified Events & Latencies: {pot_identified}\n",
            f"Average Latency: {pot_delay} timestamps\n",
            f"Unidentified Events: {pot_unidentified}\n\n\n",
            "epsilon algorithm:\n",
            f"Threshold: {e_thresh:.4f}\n",
            f"Without PA:\t F1-Score: {f1_e*100:.4f}%, \t Recall: {rec_e*100:.4f}%, \t Precision: {prec_e*100:.4f}%\n",
            f"With PA:\t {pa_f1_e*100:.4f}%, \t Recall: {pa_rec_e*100:.4f}%, \t Precision: {pa_prec_e*100:.4f}%\n",
            f"Identified Events & Latencies: {e_identified}\n",
            f"Average Latency: {e_delay} timestamps\n",
            f"Unidentified Events: {e_unidentified}\n\n\n",
            "Brute force for best F1:\n",
            f"Threshold: {bf_thresh:.4f}\n"
            f"Without PA:\t F1-Score: {f1_bf*100:.4f}%, \t Recall: {rec_bf*100:.4f}%, \t Precision: {prec_bf*100:.4f}%\n",
            f"With PA:\t {pa_f1_bf*100:.4f}%, \t Recall: {pa_rec_bf*100:.4f}%, \t Precision: {pa_prec_bf*100:.4f}%\n",
            f"Identified Events & Latencies: {bf_identified}\n",
            f"Average Latency: {bf_delay} timestamps\n",
            f"Unidentified Events: {bf_unidentified}\n\n\n",
            "Brute force for best point-adjusted F1:\n",
            f"Threshold: {pa_bf_thresh:.4f}\n"
            f"Without PA:\t F1-Score: {f1_bf_2*100:.4f}%, \t Recall: {rec_bf_2*100:.4f}%, \t Precision: {prec_bf_2*100:.4f}%\n",
            f"With PA:\t {pa_f1_bf_2*100:.4f}%, \t Recall: {pa_rec_bf_2*100:.4f}%, \t Precision: {pa_prec_bf_2*100:.4f}%\n",
            f"Identified Events & Latencies: {bf_identified_2}\n",
            f"Average Latency: {bf_delay_2} timestamps\n",
            f"Unidentified Events: {bf_unidentified_2}\n\n\n"
        ]

        f.writelines(lines_to_write)

    print("Finished.")