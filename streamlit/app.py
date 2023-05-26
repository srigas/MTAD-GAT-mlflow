import streamlit as st
import mlflow
from streamlit_utils import *
import pandas as pd

def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():

    # Set the proper mlflow tracking URI, or comment to use default
    mlflow.set_tracking_uri("file:///C:/Users/rigas/Desktop/MTAD-GAT-mlflow/mlruns")

    # Setup client to get metrics later on
    client = mlflow.MlflowClient()

    st.set_page_config(layout="wide", initial_sidebar_state="expanded")

    st.title("Anomaly Detection Model Debugging")

    st.write("""This is a custom frontend that is supposed to be used alongside MLflow's UI for model debugging and parameter tuning. While MLflow's UI provides some basic options for plotting, Anomaly Detection is a somewhat more complex problem compared to usual regression or classification and therefore requires more extensive plotting. To proceed with using this UI, select a run from the ones available in the sidebar.""")

    # the sidebar is used for parameter setting

    with st.sidebar:
        st.markdown("""## Find run for inspection""")
        dataset = st.text_input("Choose Dataset", "system_1")

        exp_name = f"{dataset}_training"
        runs_df = mlflow.search_runs(experiment_names=[exp_name])
        names = runs_df['tags.mlflow.runName']

        run_name = st.radio(
            "Choose run by name", names.values
        )

    run_id = runs_df[runs_df['tags.mlflow.runName']==run_name]['run_id'].values[0]

    st.markdown(f"Selected run with ID: {run_id}")

    # Get artifacts path for specific run_id
    art_uri = mlflow.get_run(run_id).info.artifact_uri

    # First Section
    st.markdown("### Loss Graphs")

    # Plot losses
    fig1, fig2, fig3 = plot_losses(client, run_id)

    show_losses = st.checkbox('Show loss graphs')

    if show_losses:
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
        st.plotly_chart(fig3)

    # Second Section
    st.markdown("### Anomaly Scores")

    # Load thresholds as dict
    thresholds = mlflow.artifacts.load_dict(art_uri+"/thresholds.json")

    anom_scor_opts = list(thresholds.keys())
    anom_scor_opts.append("None")

    thresh_choice = st.radio(
        "Choose threshold for anomalies",
        anom_scor_opts,
        horizontal=True
    )

    if thresh_choice != "None":
        threshold = thresholds[thresh_choice]
        figA = plot_scores(dataset, art_uri, threshold)

        st.plotly_chart(figA)

        show_correct_preds = st.checkbox('Show the scores along with the correctly predicted anomalies')

        if show_correct_preds:
            figB = plot_correct_preds(dataset, art_uri, threshold)
            st.plotly_chart(figB)


    # Third section
    st.markdown("### Feature-level Predictions")

    df_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="per_feat_scores.pkl")

    df = pd.read_pickle(df_path)

    show_feature_preds = st.checkbox('Show feature-level predictions and scores')

    if show_feature_preds:
        # Choose feature to plot
        feats = {f"Feat {f+1}" : f for f in range(df.shape[1]//4)}
        chosen_feat = st.selectbox("Choose the feature for detailed results", feats.keys())
        feat = feats[chosen_feat]

        # Choose index range
        slider_max = df.shape[0]
        (start_plot, end_plot) = st.slider("Choose a range index", min_value=0, max_value=slider_max, value=(0,slider_max))

        figC1, figC2 = plot_feat_details(dataset, df, feat, start=start_plot, end=end_plot)

        st.plotly_chart(figC1)
        st.plotly_chart(figC2)