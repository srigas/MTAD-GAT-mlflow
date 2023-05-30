## 📌 Streamlit UI

To be able to perform more detailed debugging and model tuning, this repository contains a Streamlit UI. The server connects to MLflow's tracking server's URI (hosted on Databricks) and loads all the available runs. By choosing a run by run name, a user is able to investigate detailed plots concerning the model's training & evaluation, including its losses during training, its final predictions per threshold and how they relate to the ground truth, as well as how successful the reconstruction and forecasting of each feature is.

Within the code, `mlflow.set_tracking_uri("databricks")` is set, so that the UI is connected to the databricks-hosted MLflow server. Additionally, two environment variables are set at the Web App level (through the corresponding Azure service):

```
DATABRICKS_HOST = "https://adb-XXXXXXXXX.azuredatabricks.net/"
```

where `XXXXXXXXX` is the workspace ID and

```
DATABRICKS_TOKEN = "xxxxxxxxx""
```

where `xxxxxxxxx` is a user-generated access token (to generate it, click on your Databricks Username and select User Settings). These are done to ensure that the UI is connected to the correct Databricks workspace.