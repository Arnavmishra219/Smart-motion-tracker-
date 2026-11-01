
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import av

from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SmartMotionTracker",
    page_icon="🎯",
    layout="wide"
)


# --------------------------------------------------
# YOLO VIDEO PROCESSOR
# --------------------------------------------------

class YOLOVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = self.model.track(
            img,
            persist=True,
            classes=[0],
            conf=0.4,
            tracker="bytetrack_custom.yaml",
            verbose=False
        )

        result = results[0]
        annotated_frame = result.plot()

        # Count detected people
        people_count = len(result.boxes) if result.boxes is not None else 0

        # Display people count
        cv2.putText(
            annotated_frame,
            f"People Count: {people_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )


# --------------------------------------------------
# PROJECT TITLE
# --------------------------------------------------

st.title("🎯 SmartMotionTracker")

st.subheader(
    "Human Movement Analysis using Computer Vision and Machine Learning"
)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.header("Project Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Pose Analysis",
        "ML Anomaly Detection",
        "Live YOLO Detection"
    ]
)


# --------------------------------------------------
# DATA FILES
# --------------------------------------------------

pose_file = "data/human_pose_data.csv"
ml_file = "data/ml_results.csv"


# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.header("Project Overview")

    if os.path.exists(pose_file):

        df = pd.read_csv(pose_file)

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Observations",
            len(df)
        )

        if "body_part" in df.columns:
            col2.metric(
                "Body Parts Tracked",
                df["body_part"].nunique()
            )

        st.subheader("Movement Data")

        st.dataframe(df, use_container_width=True)

    else:
        st.warning("Pose data not found. Run data_collection.py first.")


# --------------------------------------------------
# PAGE 2: POSE ANALYSIS
# --------------------------------------------------

elif page == "Pose Analysis":

    st.header("Human Pose Analysis")

    if os.path.exists(pose_file):

        df = pd.read_csv(pose_file)

        if all(col in df.columns for col in ["body_part", "x", "y"]):

            body_parts = df["body_part"].dropna().unique()

            if len(body_parts) > 0:

                selected_part = st.selectbox(
                    "Select Body Part",
                    body_parts
                )

                filtered_df = df[
                    df["body_part"] == selected_part
                ]

                st.subheader(
                    f"{selected_part} Movement Trajectory"
                )

                fig, ax = plt.subplots()

                ax.plot(
                    filtered_df["x"],
                    filtered_df["y"],
                    marker="o"
                )

                ax.set_xlabel("X Coordinate")
                ax.set_ylabel("Y Coordinate")
                ax.set_title(
                    f"{selected_part} Movement"
                )

                ax.invert_yaxis()
                ax.grid(True)

                st.pyplot(fig)
                plt.close(fig)

            else:
                st.warning("No body-part data available.")

        else:
            st.error("Required columns are missing from pose data.")

    else:
        st.warning("Pose data not found. Run data_collection.py first.")


# --------------------------------------------------
# PAGE 3: ML ANOMALY DETECTION
# --------------------------------------------------

elif page == "ML Anomaly Detection":

    st.header("ML-Based Movement Anomaly Detection")

    if os.path.exists(ml_file):

        df = pd.read_csv(ml_file)

        if "movement_status" in df.columns:

            normal_count = (
                df["movement_status"] == "Normal"
            ).sum()

            unusual_count = (
                df["movement_status"] == "Unusual"
            ).sum()

            col1, col2 = st.columns(2)

            col1.metric(
                "Normal Movements",
                int(normal_count)
            )

            col2.metric(
                "Unusual Movements",
                int(unusual_count)
            )

            st.subheader("Movement Classification")

            st.dataframe(
                df,
                use_container_width=True
            )

        if all(col in df.columns for col in
               ["time_seconds", "speed", "movement_status"]):

            st.subheader("Movement Speed Analysis")

            fig, ax = plt.subplots()

            for status in df["movement_status"].dropna().unique():

                subset = df[
                    df["movement_status"] == status
                ]

                ax.scatter(
                    subset["time_seconds"],
                    subset["speed"],
                    label=status
                )

            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Movement Speed")
            ax.set_title("Normal vs Unusual Movement")

            ax.legend()
            ax.grid(True)

            st.pyplot(fig)
            plt.close(fig)

    else:
        st.warning("ML results not found. Run human_pose_ml.py first.")


# --------------------------------------------------
# PAGE 4: LIVE YOLO DETECTION
# --------------------------------------------------

elif page == "Live YOLO Detection":

    st.header("Live Human Detection and Tracking")

    st.write(
        "Real-time human detection, people counting "
        "and multi-person tracking using YOLO and ByteTrack."
    )

    # Check required files
    if not os.path.exists("yolo11n.pt"):

        st.error("YOLO model file yolo11n.pt was not found.")

    elif not os.path.exists("bytetrack_custom.yaml"):

        st.error("bytetrack_custom.yaml was not found.")

    else:

        st.info(
            "Allow camera access in your browser when prompted. "
            "Keep a person visible to maintain their tracking ID."
        )

        RTC_CONFIGURATION = RTCConfiguration(
            {
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            }
        )

        webrtc_streamer(
            key="smart-motion-tracker",
            video_processor_factory=YOLOVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

        st.caption(
            "Note: Tracking IDs are temporary IDs for objects "
            "in the current camera stream, not permanent identities."
        )
