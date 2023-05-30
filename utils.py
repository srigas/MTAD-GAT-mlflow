import os
import numpy as np
import datetime
from torch.utils.data import DataLoader, Dataset, Subset
from spot import SPOT
import mlflow


# ------------------------ DATABRICKS UTILITIES ------------------------------

def str2bool(input_string):
    """Transform various ways of writing yes/no uniformly

    :param input_string: string to be transformed
    """
    if input_string.lower() in ("yes", "true", "t", "y", "1"):
        return "True"
    elif input_string.lower() in ("no", "false", "f", "n", "0"):
        return "False"
    else:
        return input_string


def str2type(input_string):
    """Transform an input string to the corresponding type, based on its contents

    :param input_string: string to be transformed
    """

    input_string = str2bool(input_string)

    if input_string == "True":
        output = True
    elif input_string == "False":
        output = False
    elif input_string == "None":
        output = None
    # This is a very specific case, e.g. for the "patience" or "train_end" parameters
    else:
        output = int(input_string)

    return output


# ------------------------ DATA UTILITIES ------------------------------


def get_data(dataset, mode="train", start=0, end=None):
    """Get data to be used for training/validation/evaluation

    :param mode: train, eval or new, to get train, eval or new data
    :param start: starting index of dataset if not all data are to be used
    :param end: ending index of dataset if not all data are to be used
    """
    
    dataset_folder = os.path.join("/dbfs/mnt/datasets", dataset)

    # Load the data
    # WARNING: For good evaluation/inference, a total of window_size data need to be taken
    # from the train dataset and placed before the new data. This should not be done within
    # the model, but as a different pre-processing procedure.
    try:
        data = np.loadtxt(os.path.join(dataset_folder, f"{mode}.txt"),
                            delimiter=",", dtype=np.float32)[start:end, :]
        if mode=="train":
            # train data do not have labels - unsupervised learning
            labels = None
        elif mode=="new":
            # new coming data are also not labeled
            labels = None
        else:
            labels = np.loadtxt(os.path.join(dataset_folder, "labels.txt"),
                                delimiter=",", dtype=np.float32)[start:end]
    except (KeyError, FileNotFoundError):
        raise Exception("Only acceptable modes are train/eval/new.")

    return (data, labels)


class SlidingWindowDataset(Dataset):
    """Class that creates a sliding window dataset for a given time-series

    :param data: time-series data to be converted
    :param window_size: size of the sliding window
    :param stride: the number of different timestamps between two consecutive windows
    :param horizon: the number of timestamps for future predictions
    """
    def __init__(self, data, window_size, stride=1, horizon=1):
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.horizon = horizon

    def __getitem__(self, index):
        start_index = index * self.stride
        x = self.data[start_index : start_index + self.window_size]
        y = self.data[start_index + self.window_size : start_index + self.window_size + self.horizon]
        return x, y

    def __len__(self):
        return len(range(0, len(self.data) - self.window_size - self.horizon + 1, self.stride))


def create_data_loader(dataset, batch_size, val_split=0.1, shuffle=True):
    """Create torch data loaders to feed the data in the model

    :param dataset: torch dataset
    :param batch_size: size of data batches
    :param val_split: if set to a non-zero value, an extra loader is created with val_split*100%
                      of the whole data, usually to be used for validation
    :param shuffle: wether to shuffle data and get random indices or not
    """
    if val_split is None:
        # Corresponds to the case of eval data or training without validation
        print(f"The size of the dataset is: {len(dataset)} sample(s).")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        extra_loader = None

    else:
        # Corresponds to the case with train/val splitting
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        extra_loader = DataLoader(val_dataset, batch_size=batch_size)

        print(f"The size of the dataset is: {len(train_indices)} sample(s).")
        print(f"Reserved {len(val_indices)} sample(s) for validation.")

    return loader, extra_loader


def get_run_id(run_name, experiment_name):
    """Transform an input run_name to the run_id

    :param run_name: the input run_name to be transformed
    :param experiment_name: the name of the experiment in which to look for runs
    """

    # If no run_name is given, the last run is retrieved
    if run_name is None:
        run_name = "-1"
    else:
        run_name = str(run_name)

    # get corresponding experiment (using experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id

    # If run_name is given as a relative value (-1, -2, etc.), get the actual name
    if run_name.startswith('-'):
        
        runs = mlflow.search_runs(experiment_ids=exp_id)
        run_names = runs["tags.mlflow.runName"].values.tolist()
        date_times = [datetime.datetime.strptime(rn, '%d%m%Y_%H%M%S') for rn in run_names]
        date_times.sort()
        model_datetime = date_times[int(run_name)]
        run_name = model_datetime.strftime('%d%m%Y_%H%M%S')
    
    # Given the actual name, retrieve the run id
    run = mlflow.search_runs(experiment_ids=exp_id, filter_string=f'tags."mlflow.runName" = "{run_name}"')
    run_id = run['run_id'][0]

    return run_id


# ------------------------ THRESHOLD UTILITIES ------------------------------


def pot_threshold(init_score, score, q=1e-3, level=0.99, dynamic=False):
    """
    Run POT method on given score.
    :param init_score (np.ndarray): The data to get init threshold.
                    For `OmniAnomaly`, it should be the anomaly score of train set.
    :param: score (np.ndarray): The data to run POT method.
                    For `OmniAnomaly`, it should be the anomaly score of test set.
    :param q (float): Detection level (risk)
    :param level (float): Probability associated with the initial threshold t
    :return threshold: pot result threshold
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Running POT with q={q}, level={level}..")
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)
    s.initialize(level=level, min_extrema=False)  # Calibration step
    ret = s.run(dynamic=dynamic, with_alarm=False)

    #print(f'While running POT, detected {len(ret["alarms"])} and calculated {len(ret["thresholds"])} thresholds.')

    pot_th = np.mean(ret["thresholds"])
    return pot_th


def find_epsilon(errors, reg_level=1):
    """
    Threshold method proposed by Hundman et. al. (https://arxiv.org/abs/1802.04431)
    Code from TelemAnom (https://github.com/khundman/telemanom)
    """
    e_s = errors
    best_epsilon = None
    max_score = -10000000
    mean_e_s = np.mean(e_s)
    sd_e_s = np.std(e_s)

    for z in np.arange(2.5, 12, 0.5):
        epsilon = mean_e_s + sd_e_s * z
        pruned_e_s = e_s[e_s < epsilon]

        i_anom = np.argwhere(e_s >= epsilon).reshape(-1,)
        buffer = np.arange(1, 50)
        i_anom = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) > 0:

            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
            if reg_level == 0:
                denom = 1
            elif reg_level == 1:
                denom = len(i_anom)
            elif reg_level == 2:
                denom = len(i_anom) ** 2

            score = (mean_perc_decrease + sd_perc_decrease) / denom

            if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                max_score = score
                best_epsilon = epsilon

    if best_epsilon is None:
        best_epsilon = np.max(e_s)
    return best_epsilon


def json_to_numpy(path):
    """Opens a .json artifact and casts its values as a numpy array
    :param path: path to look for the json artifact 
    """

    data = mlflow.artifacts.load_dict(path)

    npfile = np.asarray(list(data.values())).flatten()

    return npfile


def update_json(uri, name, new_data):
    """Opens a .json artifact and updates its contents with new data
    :param path: path to look for the json artifact
    :param new_data: dictionary that contains the new contents as key-value pairs    
    """
    try:
        data = mlflow.artifacts.load_dict(uri+"/"+name)
        data.update(new_data)
    except mlflow.exceptions.MlflowException:
        data = new_data

    mlflow.log_dict(data, name)


# ------------------------ EVALUATION UTILITIES ------------------------------


def get_metrics(y_pred, y_true):
    """Function to calculate metrics, given a predictions and an actual list of 0s and 1s.
    :param y_pred: list of 0s and 1s as predicted by the model
    :param y_true: list of 0s and 1s as ground truth anomalies
    """
    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)

    TP = np.sum(y_pred * y_true)
    FP = np.sum(y_pred * (1 - y_true))
    FN = np.sum((1 - y_pred) * y_true)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)

    return f1, precision, recall


def anoms_to_indices(anom_list):
    """Function that returns indices of anomalous values
    :param anom_list: list of 0s and 1s
    """
    ind_list = [i for i, x in enumerate(anom_list) if x == 1]
    xs = list(range(len(anom_list)))
    return ind_list, xs


def create_anom_range(xs, anoms):
    """Function that creates ranges of anomalies
    :param xs: list of indices to be used for the plot, auto generated by anoms_to_indices
    :param anoms: indices that belong in xs and correspond to anomalies
    """
    anomaly_ranges = []
    for anom in anoms:
        idx = xs.index(anom)
        if anomaly_ranges and anomaly_ranges[-1][-1] == idx-1:
            anomaly_ranges[-1] = (anomaly_ranges[-1][0], idx)
        else:
            anomaly_ranges.append((idx, idx))
    return anomaly_ranges


def PA(y_true, y_pred):
    """Function that performs the point-adjustment strategy
    :param y_true: list of 0s and 1s as ground truth anomalies
    :param y_pred: list of 0s and 1s as predicted by the model, so that they can be point-adjusted
    """
    new_preds = np.array(y_pred)

    # Transform into indices lists
    y_true_ind, xs = anoms_to_indices(y_true)
    y_pred_ind, _ = anoms_to_indices(y_pred)

    # Create the anomaly ranges
    anom_ranges = create_anom_range(xs, y_true_ind)

    # Iterate over all ranges
    for start, end in anom_ranges:
        itms = list(range(start,end+1))
        # if we find at least one identified instance
        if any(item in itms for item in y_pred_ind):
            # Set the whole event equal to 1
            new_preds[start:end+1] = 1

    return new_preds


def calculate_latency(y_true, y_pred):
    """Function that calculates the latency of all events' prediction
    :param y_true: list of 0s and 1s as ground truth anomalies
    :param y_pred: list of 0s and 1s as predicted by the model, so that they can be point-adjusted
    """
    events = []
    identified_events = []
    
    # Identify separate events in the ground truth
    start = None
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if start is None:
                start = i
        elif start is not None:
            events.append((start, i-1))
            start = None
    
    # Identify events and calculate delays in predictions
    for event in events:
        start, end = event
        delay = None
        for i in range(start, end+1):
            if y_pred[i] == 1:
                delay = i - start
                break
        if delay is not None:
            identified_events.append((event, delay))
    
    num_correct = len(identified_events)
    total_delay = sum(delay for _, delay in identified_events)
    avg_delay = total_delay / num_correct if num_correct > 0 else 0
    
    # Events not identified
    not_identified_events = [event for event in events if event not in [e[0] for e in identified_events]]
    
    return num_correct, avg_delay, identified_events, not_identified_events