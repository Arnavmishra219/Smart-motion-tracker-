
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="SmartMotionTracker",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 SmartMotionTracker")
st.subheader("Human Movement Analysis using Computer Vision and Machine Learning")

st.sidebar.header("Project Navigation")

page = st.sidebar.radio(
    "Select Section",
    ["Overview", "Pose Analysis", "ML Anomaly Detection"]
)

pose_file = "data/human_pose_data.csv"
ml_file = "data/ml_results.csv"

if page == "Overview":
    st.header("Project Overview")

    st.write("""
    SmartMotionTracker uses:
    - OpenCV for camera-based motion processing
    - MediaPipe for human pose estimation
    - Python for data collection
    - Scikit-learn for machine learning
    """)

    if os.path.exists(pose_file):
        df = pd.read_csv(pose_file)

        col1, col2 = st.columns(2)
        col1.metric("Total Observations", len(df))
        col2.metric("Body Parts Tracked", df["body_part"].nunique())

        st.success("Human pose data loaded successfully!")
    else:
        st.warning("Pose data not found. Run human_pose_tracker.py first.")

elif page == "Pose Analysis":
    st.header("Human Pose Trajectory Analysis")

    if os.path.exists(pose_file):
        df = pd.read_csv(pose_file)

        selected_part = st.selectbox(
            "Select Body Part",
            sorted(df["body_part"].unique())
        )

        part_data = df[df["body_part"] == selected_part]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(part_data["x"], part_data["y"])
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.invert_yaxis()
        ax.set_title(f"{selected_part} Movement")
        ax.grid(True)

        st.pyplot(fig)
    else:
        st.warning("Run human_pose_tracker.py first.")

elif page == "ML Anomaly Detection":
    st.header("Machine Learning Results")

    if os.path.exists(ml_file):
        df = pd.read_csv(ml_file)

        st.write("### Movement Classification")
        st.dataframe(df[[
            "time_seconds",
            "body_part",
            "speed",
            "movement_status"
        ]].tail(100))

        normal = (df["anomaly"] == 1).sum()
        unusual = (df["anomaly"] == -1).sum()

        col1, col2 = st.columns(2)
        col1.metric("Normal Observations", int(normal))
        col2.metric("Unusual Observations", int(unusual))

        fig, ax = plt.subplots(figsize=(9, 5))

        normal_data = df[df["anomaly"] == 1]
        unusual_data = df[df["anomaly"] == -1]

        ax.scatter(
            normal_data["time_seconds"],
            normal_data["speed"],
            label="Normal",
            s=12
        )

        ax.scatter(
            unusual_data["time_seconds"],
            unusual_data["speed"],
            label="Unusual",
            marker="x",
            s=20
        )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Speed (pixels/second)")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    else:
        st.warning("Run human_pose_ml.py first.")
