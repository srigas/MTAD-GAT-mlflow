import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import mlflow
import os

def vals_from_metric(client, run_id, key):
    metrics = client.get_metric_history(run_id=run_id, key=key)
    vals = [m.value for m in metrics]
    return np.asarray(vals)

def get_losses(client, run_id):
    fc_train = vals_from_metric(client, run_id, "train_fc_loss")
    recon_train = vals_from_metric(client, run_id, "train_rc_loss")

    fc_val = vals_from_metric(client, run_id, "val_fc_loss")
    recon_val = vals_from_metric(client, run_id, "val_rc_loss")

    tot_train = vals_from_metric(client, run_id, "total_train_loss")
    tot_val = vals_from_metric(client, run_id, "total_val_loss")

    return fc_train, recon_train, fc_val, recon_val, tot_train, tot_val

def plot_losses(client, run_id):

    fc_train, recon_train, fc_val, recon_val, tot_train, tot_val = get_losses(client, run_id)

    epochs = list(range(1,fc_train.shape[0]+1))
    
    cust_cols = {'total_train' : '#003f5c', 'total_val' : '#d45087', 'forecast' : '#665191',
                 'recon' : '#ffa600', 'patience' : '#ffa600'}

    # Create the subplots
    fig1 = go.Figure()
    fig2 = go.Figure()
    fig3 = go.Figure()

    # Add traces to the first graph
    fig1.add_trace(go.Scatter(x=epochs, y=fc_train, mode='lines', name='Forecasting', line=dict(color=cust_cols['forecast'], width=2.0)))
    fig1.add_trace(go.Scatter(x=epochs, y=recon_train, mode='lines', name='Reconstruction', line=dict(color=cust_cols['recon'], width=2.0)))
    fig1.update_layout(title="Training losses during model's training", xaxis_title="Epoch", yaxis_title="Loss")

    # Add traces to the second graph
    fig2.add_trace(go.Scatter(x=epochs, y=fc_val, mode='lines', name='Forecasting', line=dict(color=cust_cols['forecast'], width=2.0)))
    fig2.add_trace(go.Scatter(x=epochs, y=recon_val, mode='lines', name='Reconstruction', line=dict(color=cust_cols['recon'], width=2.0)))
    fig2.update_layout(title="Validation losses during model's training", xaxis_title="Epoch", yaxis_title="Loss")

    # Add traces to the third graph
    fig3.add_trace(go.Scatter(x=epochs, y=tot_train, mode='lines', name='Training', line=dict(color=cust_cols['total_train'], width=2.0)))
    fig3.add_trace(go.Scatter(x=epochs, y=tot_val, mode='lines', name='Validation', line=dict(color=cust_cols['total_val'], width=2.0)))
    fig3.add_shape(type="line", x0=epochs[tot_val.argmin()], y0=min(tot_val.min(), tot_train.min()), x1=epochs[tot_val.argmin()], y1=max(tot_val.max(), tot_train.max()),
                line=dict(color=cust_cols['patience'], dash='dash'))
    fig3.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Patience limit', line=dict(color=cust_cols['patience'], dash='dash')))
    fig3.update_layout(title="Training vs Validation losses during model's training", xaxis_title="Epoch", yaxis_title="Loss")
    
    fig1.update_layout(height=500, width=800)
    fig2.update_layout(height=500, width=800)
    fig3.update_layout(height=500, width=800)

    return fig1, fig2, fig3

def get_labels(art_uri):

    labels = json_to_numpy(art_uri+"/labels.json")

    return labels

def plot_scores(art_uri, threshold):

    # Load scores
    scores = json_to_numpy(art_uri+"/eval_scores.json")

    # load ground truth
    labels = get_labels(art_uri)

    # Get actual predictions given the scores and threshold
    anoms = np.asarray([0 if score < threshold else 1 for score in scores])
    
    # Perform point-adjustment
    pa_anoms = PA(labels, anoms)
    
    indices = list(range(labels.shape[0]))
    cust_cols = {'scores' : '#665191', 'threshold' : '#ff7c43', 'preds' : '#f95d6a', 'preds-pa' : '#60C689', 'truth' : '#2f4b7c'}

    # Create the subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.4, 0.1, 0.1, 0.1])

    # Add traces to the subplots
    fig.add_trace(go.Scatter(x=indices, y=scores, mode='lines', name='Anomaly Scores', line=dict(color=cust_cols['scores'], width=1.0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=indices, y=[threshold] * len(indices), mode='lines', name='Threshold', line=dict(color=cust_cols['threshold'], dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=indices, y=anoms, mode='lines', name='Predicted Anomalies', line=dict(color=cust_cols['preds'], width=1.0)), row=2, col=1)
    fig.add_trace(go.Scatter(x=indices, y=pa_anoms, mode='lines', name='Adjusted Anomalies', line=dict(color=cust_cols['preds-pa'], width=1.0)), row=3, col=1)
    fig.add_trace(go.Scatter(x=indices, y=labels, mode='lines', name='Ground Truth', line=dict(color=cust_cols['truth'], width=1.0)), row=4, col=1)

    # Update the layout for each subplot
    fig.update_layout(
        height=700,
        title_text=f'Anomaly Scores for threshold: {threshold:.4f}',
        showlegend=True,
        margin=dict(t=150)
    )

    # Add the legend at the bottom
    fig.update_layout(
        legend=dict(
            x=0,
            y=1.1,
            xanchor="left",
            orientation="h",
            itemsizing='constant'
        )
    )

    return fig

def plot_correct_preds(art_uri, threshold):

    # Load scores
    scores = json_to_numpy(art_uri+"/eval_scores.json")

    # load ground truth
    labels = get_labels(art_uri)

    # Get actual predictions given the scores and threshold
    anoms = np.asarray([0 if score < threshold else 1 for score in scores])
    
    # Perform point-adjustment
    pa_anoms = PA(labels, anoms)
    # Get correct predictions
    found = labels*pa_anoms 
    # Make ranges
    inds, xs = anoms_to_indices(found)
    found = create_anom_range(xs, inds)
    
    indices = list(range(labels.shape[0]))

    cust_cols = {'scores' : '#665191', 'anoms' : '#E91E63', 'threshold' : '#ff7c43'}

    # Create the figure
    fig = go.Figure()

    # Add traces to the figure
    fig.add_trace(go.Scatter(x=indices, y=scores, mode='lines', name='Anomaly Scores', line=dict(color=cust_cols['scores'], width=1.0)))
    fig.add_trace(go.Scatter(x=indices, y=[threshold] * len(indices), mode='lines', name='Threshold', line=dict(color=cust_cols['threshold'], dash='dash')))
    
    # Add a transparent scatter trace for identified anomalies to include in the legend
    fig.add_trace(go.Scatter(x=indices, y=[None] * len(indices), mode='lines', line=dict(color=cust_cols['anoms']), name='Identified Anomalies'))

    # Add correctly identified anomalies
    for idx, (start, end) in enumerate(found):
        fig.add_shape(type="rect", x0=start, y0=0, x1=end, y1=1.05*scores.max(), fillcolor=cust_cols['anoms'], opacity=0.2, line=dict(color='rgba(0, 0, 0, 0)'))

    # Update the layout
    fig.update_layout(
        xaxis=dict(title="Indices"),
        yaxis=dict(title="Anomaly Scores"),
        showlegend=True,
        legend=dict(
            x=0,
            y=1.1,
            xanchor="left",
            orientation="h",
            itemsizing='constant'
        )
    )

    return fig

def plot_feat_details(art_uri, df, feat, start=0, end=None):
    fcs = df[f'FC_{feat}'].values[start:end]
    recons = df[f'RECON_{feat}'].values[start:end]
    actual = df[f'TRUE_{feat}'].values[start:end]
    scores = df[f'SCORE_{feat}'].values[start:end]

    # load ground truth
    labels = get_labels(art_uri)

    # Make ranges for actual anomalies
    inds, xs = anoms_to_indices(labels)
    anoms = create_anom_range(xs, inds)

    indices = list(range(labels.shape[0]))[start:end]
    cust_cols = {'scores' : '#665191', 'fcs' : '#60C689', 'recons' : '#ff7c43', 'actual' : '#2f4b7c', 'anoms' : '#E91E63'}
    
    # Create the first figure for predictions and measurements
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=indices, y=fcs, mode='lines', name='Forecasting', line=dict(color=cust_cols['fcs'], width=1.0)))
    fig1.add_trace(go.Scatter(x=indices, y=recons, mode='lines', name='Reconstruction', line=dict(color=cust_cols['recons'], width=1.0)))
    fig1.add_trace(go.Scatter(x=indices, y=actual, mode='lines', name='Measured', line=dict(color=cust_cols['actual'], width=1.0)))

    # Add a transparent scatter trace for identified anomalies to include in the legend
    fig1.add_trace(go.Scatter(x=indices, y=[None] * len(indices), mode='lines', line=dict(color=cust_cols['anoms']), name='Identified Anomalies'))

    # Add ground truth anomalies to the first figure
    for (a, b) in anoms:
        if a >= start and (end is None or b <= end):
            fig1.add_shape(type="rect", x0=a, y0=min(actual.min(), recons.min(), fcs.min()), x1=b, y1=max(actual.max(), recons.max(), fcs.max()), fillcolor=cust_cols['anoms'], opacity=0.2, line=dict(color='rgba(0, 0, 0, 0)'))

    fig1.update_layout(title=f'Predictions and Measurements for Feature No. {feat+1}', xaxis_title="Indices", yaxis_title="Value", showlegend=True,
                       legend=dict(x=0, y=1.1, xanchor="left", orientation="h", itemsizing='constant'))

    # Create the second figure for anomaly scores
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=indices, y=scores, mode='lines', name='Scores', line=dict(color=cust_cols['scores'], width=1.0)))

    # Add a transparent scatter trace for identified anomalies to include in the legend
    fig2.add_trace(go.Scatter(x=indices, y=[None] * len(indices), mode='lines', line=dict(color=cust_cols['anoms']), name='Identified Anomalies'))

    # Add ground truth anomalies to the second figure
    for (a, b) in anoms:
        if a >= start and (end is None or b <= end):
            fig2.add_shape(type="rect", x0=a, y0=min(scores), x1=b, y1=max(scores), fillcolor=cust_cols['anoms'], opacity=0.2, line=dict(color='rgba(0, 0, 0, 0)'))

    fig2.update_layout(title=f'Anomaly Scores for Feature No. {feat+1}', xaxis_title="Indices", yaxis_title="Score", showlegend=True, legend=dict(x=0, y=1.1, xanchor="left", orientation="h", itemsizing='constant'))
    
    return fig1, fig2

# ----------- got these from original utils file ------------------
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

def json_to_numpy(path):
    """Opens a .json artifact and casts its values as a numpy array
    :param path: path to look for the json artifact 
    """

    data = mlflow.artifacts.load_dict(path)

    npfile = np.asarray(list(data.values())).flatten()

    return npfile
# -----------------------------------------------------------------
