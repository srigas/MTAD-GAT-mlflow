# MTAD-GAT meets MLflow

The present repository contains modularized code and files related to the [MTAD-GAT model](https://github.com/ML4ITS/mtad-gat-pytorch) for multivariate time-series anomaly detection, written using MLflow's Tracking and Model components. Apart from MLflow's built-in UI, code for an additional UI written in Streamlit is provided, to be used for detailed model debugging and tuning.

## 📌 MTAD-GAT

The main directory contains the code for the MTAD-GAT model. To run the code, create a new conda environment using

```
conda create --name ENV_NAME python==3.8.11
```

and subsequently install all dependencies using

```
pip3 install -r requirements.txt
```

The code contains three main scripts:

• [train.py](/train.py): Run using `python train.py --arg1 value1 --arg2 value2 ...` so that the training of the model on the available training data (see [train.txt](/datasets/system_1/train.txt)) is initiated. The possible arguments can be found in the `train_parser()` function of the [args.py](/args.py) file.

• [evaluate.py](/evaluate.py): Run using `python evaluate.py --arg1 value1 --arg2 value2 ...` so that the evaluation of the model on the available evaluation data (see [eval.txt](/datasets/system_1/eval.txt) and [labels.txt](/datasets/system_1/labels.txt)) is initiated. The possible arguments can be found in the `predict_parser()` function of the [args.py](/args.py) file.

• [predict.py](/predict.py): Run using `python predict.py --arg1 value1 --arg2 value2 ...` so that inference on the required prediction data (see [new.txt](/datasets/system_1/new.txt)) is initiated. The possible arguments can be found in the `predict_parser()` function of the [args.py](/args.py) file.

The code is organized so that it is as little model-dependent as possible. The [architecture.py](/architecture.py) file contains all the model-specific modules corresponding to the model and its architecture, while the [model.py](/model.py) file contains all the routines required for the training, evaluation and scoring of the model and is somewhat model-specific. This implies that these are the main two files that need to be tampered with in case another model is to be used in the future. The [utils.py](/utils.py) file contains some generic utilities for model training, evaluation, as well as thresholding.

## 📌 Thresholding

The purpose of an Anomaly Detection model is to yield anomaly scores for every timestamp contained in a time-series which is inserted for inference. Regardless of the underlying model, proper thresholding methods need to be applied so that an optimal threshold is chosen in order to differentiate between anomalous and non-anomalous points. In the present implementation, four different thresholding approaches are followed:

• Peaks Over Thresholds (POT) (see [spot.py](/spot.py) file, as well as the `pot_threshold()` function in the utilities file), as explained in [this](https://hal.science/hal-01640325/document) paper.

• Dynamic Error Thresholding (epsilon) (see the `find_epsilon` function in the utilities file), as explained in [this](https://arxiv.org/pdf/1802.04431.pdf) paper.

• Brute Force F1 (BF-F1), which is a brute-force method of finding the threshold that maximizes the F1 Score of the model's predictions on the evaluation dataset.

• Brute Force on Point-Adjusted F1 (BF-F1-PA), which is a brute-force method of finding the threshold that maximizes the Point-Adjusted F1 Score of the model's predictions on the evaluation dataset.

## 📌 MLflow

In order to be able to integrate the model into a broader Data Pipeline and ensure good MLOps practices, MLflow's [Tracking](https://mlflow.org/docs/latest/tracking.html), [Models](https://mlflow.org/docs/latest/models.html) and [Model Registry](https://mlflow.org/docs/latest/model-registry.html) components are used.

Two types of experiments are introduced per dataset: training experiments and inference experiments. Training experiments contain the runs created when running the `train.py` or `evaluate.py` scripts, while inference experiments contain the runs created when running the `predict.py` script. Note that during evaluation of a trained model a new run is **not** initiated; instead, the run that created the train model during training is continued.

During a training experiment's runs, a series of parameters, metrics and artifacts are logged throughout the model's training and subsequent evaluation, including the trained model itself. The models are logged and their performance can be evaluated through MLflow's built-in UI, by comparing results with different parameter configurations. Models that satisfy certain conditions can be registered and pushed to different stages. During an inference experiment's runs, only models at `Production` stage are used (unless none exists, in which case the most recently trained available model is used).

## 📌 Streamlit UI

To be able to perform more detailed debugging and model tuning, this repository also contains a Streamlit UI (see [here](/streamlit)). The server connects to MLflow's tracking server's URI and loads all the available runs. By choosing a run by run name, a user is able to investigate detailed plots concerning the model's training & evaluation, including its losses during training, its final predictions per threshold and how they relate to the ground truth, as well as how successful the reconstruction and forecasting of each feature is. To run the streamlit UI locally simply log into Streamlit's directory, install the requirements and after setting up the proper tracking URI in the [app.py](/streamlit/app.py) file run

```
streamlit run app.py
```

Note that the [Visualizations](/Visualizations) folder also contains similar details in the form of a `.ipynb` notebook, however it cannot provide the interactivity and automation provided by the Streamlit UI.
