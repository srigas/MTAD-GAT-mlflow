# MTAD-GAT meets MLflow (Cloud Version)

This is the `cloud` branch of the original repository, containing the revised code so that it can be run in a Databricks environment through external Databricks jobs.

## 📌 Differences

• There is no `requirements.txt` file in the parent directory. This is because the code is expected to be run in a Databricks environment using a ML cluster, where all required libraries are pre-installed.

• There is no `args.txt` file. This is because the code is not completely modularized as in the original version. Instead, the four `.py` files are `architecture.py`, `model.py`, `spot.py` and `utils.py` and act as a mini MTAD-GAT library. The remaining three files, `train`, `evaluate` and `predict` are databricks files, which take arguments through `dbutils.widgets` instead of a console and can be run through external notebooks as well.

• The data files are no longer local. Instead, they are uploaded in an ADLS2 container, where they are supposed to end up in the fully automated version of the data pipeline too. The connectivity between Databricks and ADLS2 is achieved by mounting the corresponding container in the workspace using `dbutils.fs.mount`. The authentication for this process is done by creating an application through Azure AD and then granting it the Blob Storage Data Contributor role by following [this](https://learn.microsoft.com/en-us/azure/databricks/storage/aad-storage-service-principal) procedure. Subsequenltly, the app's client secret, application (client) ID and directory (tenant) ID are added as secrets inside an Azure Key Vault, so that they can be invoked inside a Databricks notebook with a secret scope connected to this specific vault. The following links might be helpful for these:

- [Mounting containers on Databricks](https://learn.microsoft.com/en-us/azure/databricks/dbfs/mounts)
- [Ways of connecting to ADLS2](https://learn.microsoft.com/en-us/azure/databricks/storage/azure-storage)
- [Adding Secret Scopes inside Databricks](https://learn.microsoft.com/en-us/azure/databricks/security/secrets/secret-scopes)

• To run the databricks files through external files (e.g. when preparing an experiment that involves several runs), one needs to use the `dbutils.notebook.run` command, which takes as arguments the notebook's path, a timeout limit (set it to 0 to ignore), as well as an `arguments` parameter, which takes a dictionary as argument, container key-value pairs of the databricks file's parameters (similarly to calling them through the command line). Note here that the following lines may need to be inserted at the start of this external file, so that there are no issues with files that cannot be imported when running the code:

```
import sys
sys.path.append("/Workspace/MTAD-GAT")
```
