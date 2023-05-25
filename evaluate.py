import argparse
import numpy as np
import pandas as pd
import torch
import os

from args import predict_parser
from model import Handler
from utils import get_run_id, get_data, SlidingWindowDataset, create_data_loader
from utils import pot_threshold, json_to_numpy, update_json
from utils import get_metrics, PA, calculate_latency

import mlflow

if __name__ == "__main__":

    # Get arguments from console
    parser = predict_parser()
    args = parser.parse_args()

    dataset = args.dataset

    run_id = get_run_id(args.run_name, f"{dataset}_training")

    with mlflow.start_run(run_id=run_id):

        art_uri = mlflow.get_artifact_uri()

        # Get configs used for model training
        print(f'Using model from run with ID: {run_id}')
        model_parser = argparse.ArgumentParser()
        model_args, unknown = model_parser.parse_known_args()
        
        model_args.__dict__ = mlflow.artifacts.load_dict(art_uri+"/config.txt")

        window_size = model_args.window_size

        # --------------------------- START EVALUATION -----------------------------
        # Get data from the dataset
        (x_eval, y_eval) = get_data(dataset, mode="eval", start=args.eval_start, end=args.eval_end)

        # This workaround needs to happen internally at the moment
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
        eval_dataset = SlidingWindowDataset(x_eval, window_size)

        print("Evaluating:")
        # Create the data loader - no shuffling here
        eval_loader, _ = create_data_loader(eval_dataset, model_args.batch_size, None, False)

        # Load the model
        model = mlflow.pytorch.load_model(f"{art_uri}/{dataset}_model")

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

        # Get new scores
        print("Calculating scores on evaluation data...")
        new_scores, df = handler.score(loader=eval_loader, details=args.show_details)

        # Save a dataframe with per-feature predictions and scores for debugging
        if args.show_details:
            df.to_pickle('per_feat_scores.pkl')
            mlflow.log_artifact('per_feat_scores.pkl')
            os.remove('per_feat_scores.pkl')

            mlflow.log_dict({'eval_scores':new_scores.tolist()}, "eval_scores.json")

        # --> POT threshold
        
        print("Loading scores from data used for training...")
        
        train_scores = json_to_numpy(art_uri+"/anom_scores.json")

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

        update_json(art_uri, "thresholds.json", {"POT": pot_thresh})

        mlflow.log_metric(key="POT_F1", value=f1_pot)
        mlflow.log_metric(key="POT_F1-PA", value=pa_f1_pot)

        # --> epsilon threshold

        thresholds = mlflow.artifacts.load_dict(art_uri+"/thresholds.json")
        e_thresh = thresholds["epsilon"]

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

        mlflow.log_metric(key="Epsilon_F1", value=f1_e)
        mlflow.log_metric(key="Epsilon_F1-PA", value=pa_f1_e)

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

        update_json(art_uri, "thresholds.json", {"BF-F1": bf_thresh})

        mlflow.log_metric(key="Brute_Force_F1", value=f1_bf)
        mlflow.log_metric(key="Brute_Force_F1-PA", value=pa_f1_bf)
        
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

        update_json(art_uri, "thresholds.json", {"BF-F1-PA": pa_bf_thresh})

        mlflow.log_metric(key="PA_Brute_Force_F1", value=f1_bf_2)
        mlflow.log_metric(key="PA_Brute_Force_F1-PA", value=pa_f1_bf_2)

        # ---------------------------- END EVALUATION ------------------------------

        # save results
        with open('eval_summary.txt', 'w') as f:

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

        mlflow.log_artifact('eval_summary.txt')
        os.remove('eval_summary.txt')

    print("Finished.")