from collections import deque
from statistics import median
import streamlit as st
import cv2
import av
import time
import math
import threading
import pandas as pd
import mediapipe as mp

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="SmartMotionTracker",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 SmartMotionTracker")
st.caption("AI-Based Human Pose Analysis")

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
})

# ==================================================
# MEDIAPIPE SETUP
# ==================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(p1, p2):

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    angle = math.degrees(math.atan2(dy, dx))

    # Wrap to -90..+90 so a level line reads about 0 degrees
    return ((angle + 90) % 180) - 90


# ==================================================
# POSE VIDEO PROCESSOR
# ==================================================

class PoseVideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.head_angles = deque(maxlen=5)
        self.shoulder_angles = deque(maxlen=5)

        self.start_time = None
        self.last_frame_time = None
        self.fps = 0.0

        self.data_lock = threading.Lock()

        self.data = {
            "head_tilt": None,
            "shoulder_tilt": None,
            "left_wrist": "Not detected",
            "right_wrist": "Not detected",
            "detected": False,
            "history": [],
            "records": []
        }

    def get_data(self):

        with self.data_lock:

            return {
                "head_tilt": self.data["head_tilt"],
                "shoulder_tilt": self.data["shoulder_tilt"],
                "left_wrist": self.data["left_wrist"],
                "right_wrist": self.data["right_wrist"],
                "detected": self.data["detected"],
                "history": self.data["history"].copy(),
                "records": self.data["records"].copy()
            }

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        height, width = img.shape[:2]

        # Measured frames per second (smoothed)
        now = time.time()

        if self.last_frame_time is not None:
            inst = 1 / max(now - self.last_frame_time, 1e-6)
            self.fps = 0.9 * self.fps + 0.1 * inst if self.fps else inst

        self.last_frame_time = now

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            def get_point(name):

                point = landmarks[
                    mp_pose.PoseLandmark[name].value
                ]

                if point.visibility < 0.5:
                    return None

                return (
                    int(point.x * width),
                    int(point.y * height)
                )

            left_shoulder = get_point("LEFT_SHOULDER")
            right_shoulder = get_point("RIGHT_SHOULDER")

            left_wrist = get_point("LEFT_WRIST")
            right_wrist = get_point("RIGHT_WRIST")

            left_ear = get_point("LEFT_EAR")
            right_ear = get_point("RIGHT_EAR")

            shoulder_angle = None

            if left_shoulder and right_shoulder:

                shoulder_angle = calculate_angle(
                    left_shoulder,
                    right_shoulder
                )

            head_angle = None

            if left_ear and right_ear:

                head_angle = calculate_angle(
                    left_ear,
                    right_ear
                )

            if head_angle is not None:
                self.head_angles.append(head_angle)

            if shoulder_angle is not None:
                self.shoulder_angles.append(shoulder_angle)

            if self.head_angles:
                head_angle = float(median(self.head_angles))

            if self.shoulder_angles:
                shoulder_angle = float(median(self.shoulder_angles))

            # Mirrored webcam wrist mapping

            left_status = "Not detected"
            right_status = "Not detected"

            if right_wrist and right_shoulder:

                left_status = (
                    "Above"
                    if right_wrist[1] < right_shoulder[1]
                    else "Below"
                )

            if left_wrist and left_shoulder:

                right_status = (
                    "Above"
                    if left_wrist[1] < left_shoulder[1]
                    else "Below"
                )

            display_values = [

                f"Head Tilt: {head_angle:.1f} deg"
                if head_angle is not None
                else "Head Tilt: Not detected",

                f"Shoulder Tilt: {shoulder_angle:.1f} deg"
                if shoulder_angle is not None
                else "Shoulder Tilt: Not detected",

                f"Left Wrist: {left_status}",
                f"Right Wrist: {right_status}"
            ]

            for i, text in enumerate(display_values):

                cv2.putText(
                    img,
                    text,
                    (15, 30 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            # Time in seconds since this session started (starts at 0)
            current_time = time.time()

            if self.start_time is None:
                self.start_time = current_time

            elapsed = round(current_time - self.start_time, 2)

            with self.data_lock:

                self.data["head_tilt"] = head_angle
                self.data["shoulder_tilt"] = shoulder_angle

                self.data["left_wrist"] = left_status
                self.data["right_wrist"] = right_status

                self.data["detected"] = True

                self.data["history"].append({
                    "time": elapsed,
                    "head_tilt": head_angle,
                    "shoulder_tilt": shoulder_angle
                })

                self.data["history"] = (
                    self.data["history"][-150:]
                )

                self.data["records"].append({
                    "time_seconds": elapsed,
                    "head_tilt": None if head_angle is None else round(head_angle, 2),
                    "shoulder_tilt": None if shoulder_angle is None else round(shoulder_angle, 2),
                    "left_wrist": left_status,
                    "right_wrist": right_status
                })

        else:

            with self.data_lock:
                self.data["detected"] = False

            cv2.putText(
                img,
                "No person detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.putText(
            img,
            f"FPS: {self.fps:.0f}",
            (width - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ==================================================
# POSE ANALYSIS SECTION
# ==================================================

st.header("🎯 Human Pose Analysis")

st.write(
    "Live single-person pose detection and movement analysis using MediaPipe."
)

left_column, right_column = st.columns([1.6, 1])

with left_column:

    st.subheader("Live Webcam")

    pose_ctx = webrtc_streamer(
        key="pose-analysis",
        video_processor_factory=PoseVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
        async_processing=True
    )

with right_column:

    st.subheader("Live Measurements")

    head_metric = st.empty()
    shoulder_metric = st.empty()
    left_wrist_metric = st.empty()
    right_wrist_metric = st.empty()
    detection_status = st.empty()

st.divider()

st.subheader("Movement Analysis")

dashboard_placeholder = st.empty()


@st.fragment(run_every="1s")
def update_pose_dashboard():

    processor = pose_ctx.video_processor

    if processor is None:

        head_metric.metric("Head Tilt", "N/A")
        shoulder_metric.metric("Shoulder Tilt", "N/A")
        left_wrist_metric.metric("Left Wrist", "N/A")
        right_wrist_metric.metric("Right Wrist", "N/A")

        detection_status.warning(
            "Start the webcam to begin pose analysis."
        )

        with dashboard_placeholder.container():
            st.info("Movement graphs will appear when pose detection starts.")

        return

    data = processor.get_data()

    detected = data["detected"]

    head = data["head_tilt"]
    shoulder = data["shoulder_tilt"]

    left = data["left_wrist"]
    right = data["right_wrist"]

    history = data["history"]
    records = data["records"]

    # Keep a copy so the CSV can still be downloaded after Stop
    if records:
        st.session_state["last_records"] = records

    if detected:

        head_metric.metric(
            "Head Tilt",
            f"{head:.1f}°" if head is not None else "N/A"
        )

        shoulder_metric.metric(
            "Shoulder Tilt",
            f"{shoulder:.1f}°" if shoulder is not None else "N/A"
        )

        left_wrist_metric.metric("Left Wrist", left)
        right_wrist_metric.metric("Right Wrist", right)

        detection_status.success("Person detected")

    else:

        head_metric.metric("Head Tilt", "N/A")
        shoulder_metric.metric("Shoulder Tilt", "N/A")

        left_wrist_metric.metric("Left Wrist", "N/A")
        right_wrist_metric.metric("Right Wrist", "N/A")

        detection_status.warning("No person detected")

    with dashboard_placeholder.container():

        if history:

            graph_df = pd.DataFrame(history)
            graph_df = graph_df.set_index("time")

            graph_col1, graph_col2 = st.columns(2)

            with graph_col1:

                st.markdown("**Head Tilt Trend**")
                st.line_chart(graph_df[["head_tilt"]])

            with graph_col2:

                st.markdown("**Shoulder Tilt Trend**")
                st.line_chart(graph_df[["shoulder_tilt"]])

        else:

            st.info(
                "Movement graphs will appear when pose detection starts."
            )

        st.divider()
        st.subheader("Session Summary")

        if records:

            report_df = pd.DataFrame(records)

            head_values = report_df["head_tilt"].dropna()
            shoulder_values = report_df["shoulder_tilt"].dropna()

            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:

                st.markdown("**Head Tilt Statistics**")

                if not head_values.empty:

                    st.metric("Average", f"{head_values.mean():.2f}°")
                    st.metric("Minimum", f"{head_values.min():.2f}°")
                    st.metric("Maximum", f"{head_values.max():.2f}°")

                else:
                    st.info("No head tilt data available.")

            with summary_col2:

                st.markdown("**Shoulder Tilt Statistics**")

                if not shoulder_values.empty:

                    st.metric("Average", f"{shoulder_values.mean():.2f}°")
                    st.metric("Minimum", f"{shoulder_values.min():.2f}°")
                    st.metric("Maximum", f"{shoulder_values.max():.2f}°")

                else:
                    st.info("No shoulder tilt data available.")

            st.download_button(
                label="Download Pose Analysis CSV",
                data=report_df.to_csv(index=False),
                file_name="pose_analysis_report.csv",
                mime="text/csv",
                key="pose_summary_download"
            )

            duration = report_df["time_seconds"].iloc[-1]
            avg_fps = len(report_df) / duration if duration > 0 else 0

            st.caption(
                f"Total recorded frames: {len(report_df)} | "
                f"Average: {avg_fps:.1f} fps"
            )

        else:

            st.info(
                "Start the webcam to collect session data."
            )


update_pose_dashboard()

# ==================================================
# LAST SESSION REPORT (available after Stop)
# ==================================================

if not pose_ctx.state.playing and st.session_state.get("last_records"):

    st.divider()
    st.subheader("Last Session Report")

    saved_df = pd.DataFrame(st.session_state["last_records"])

    st.download_button(
        label="Download Pose Analysis CSV",
        data=saved_df.to_csv(index=False),
        file_name="pose_analysis_report.csv",
        mime="text/csv",
        key="pose_saved_download"
    )

    st.caption(f"Total recorded frames: {len(saved_df)}")

    if st.button("Clear saved record"):
        st.session_state.pop("last_records", None)
        st.rerun()
