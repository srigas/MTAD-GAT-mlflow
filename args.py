import argparse

# Converter for console-parsed arguments
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def train_parser():
    parser = argparse.ArgumentParser()

    # --- Data params ---
    parser.add_argument("--dataset", type=str, default="system_1")
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_end", type=int, default=None)

    # --- Model params ---
    # 1D conv layer
    parser.add_argument("--kernel_size", type=int, default=7)
    # GAT layers
    parser.add_argument("--use_gatv2", type=str2bool, default=True)
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
    parser.add_argument("--time_gat_embed_dim", type=int, default=None)
    # GRU layer
    parser.add_argument("--gru_n_layers", type=int, default=1)
    parser.add_argument("--gru_hid_dim", type=int, default=150)
    # Forecasting Model
    parser.add_argument("--fc_n_layers", type=int, default=3)
    parser.add_argument("--fc_hid_dim", type=int, default=150)
    # Reconstruction Model
    parser.add_argument("--recon_n_layers", type=int, default=1)
    parser.add_argument("--recon_hid_dim", type=int, default=150)
    # Other
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)

    # --- Train params ---
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--step_lr", type=int, default=10)
    parser.add_argument("--gamma_lr", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--print_every", type=int, default=1)
    # For epsilon
    parser.add_argument("--reg_level", default=1,
                        help="Controls the reg_level argument of the epsilon thresholding method. Set to None if you don't want to calculate this threshold during training.")
    parser.add_argument("--use_mov_av", type=str2bool, default=False)

    return parser


def predict_parser():
    parser = argparse.ArgumentParser()

    # --- Data params ---
    parser.add_argument("--dataset", type=str, default="system_1")
    parser.add_argument("--eval_start", type=int, default=0)
    parser.add_argument("--eval_end", type=int, default=None)

    # --- Model params ---
    parser.add_argument("--model_id", type=str, default="-1",
                        help="ID (datetime) of pretrained model to use, or -1, -2, etc. to use last, previous from last, etc. model.")

    # --- Predict params ---
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--show_details", type=str2bool, default=True)
    parser.add_argument("--threshold", type=str, default="POT")
    # If threshold is set to POT, these are the POT params
    parser.add_argument("--use_mov_av", type=str2bool, default=False)
    parser.add_argument("--q", type=float, default=1e-3)
    parser.add_argument("--level", type=float, default=0.99)
    parser.add_argument("--dynamic_pot", type=str2bool, default=False)

    return parser
