
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os

# Load collected human pose data
file_path = "data/human_pose_data.csv"

if not os.path.exists(file_path):
    print("Pose data not found! Run human_pose_tracker.py first.")
    exit()

df = pd.read_csv(file_path)

if df.empty:
    print("No data available!")
    exit()

# Calculate speed separately for each body part
df = df.sort_values(["body_part", "time_seconds"])

df["dx"] = df.groupby("body_part")["x"].diff()
df["dy"] = df.groupby("body_part")["y"].diff()
df["dt"] = df.groupby("body_part")["time_seconds"].diff()

df["speed"] = np.sqrt(df["dx"]**2 + df["dy"]**2) / df["dt"]

# Remove invalid values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["speed"])

if len(df) < 10:
    print("Not enough movement data. Collect more data first.")
    exit()

# Train Isolation Forest
features = df[["speed", "visibility"]]

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

df["anomaly"] = model.fit_predict(features)

# -1 = unusual movement, 1 = normal movement
df["movement_status"] = df["anomaly"].map({
    1: "Normal",
    -1: "Unusual"
})

print("\nMACHINE LEARNING RESULTS")
print("------------------------")
print(df["movement_status"].value_counts())

# Save results
os.makedirs("data", exist_ok=True)
df.to_csv("data/ml_results.csv", index=False)

# Plot results
plt.figure(figsize=(10, 5))

normal = df[df["anomaly"] == 1]
unusual = df[df["anomaly"] == -1]

plt.scatter(normal["time_seconds"], normal["speed"],
            label="Normal Movement", s=12)

plt.scatter(unusual["time_seconds"], unusual["speed"],
            label="Unusual Movement", s=20, marker="x")

plt.xlabel("Time (seconds)")
plt.ylabel("Speed (pixels/second)")
plt.title("Human Movement Anomaly Detection")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("data/anomaly_detection.png", dpi=300)
plt.show()

print("\nML analysis completed!")
print("Results saved in data/ml_results.csv")
print("Graph saved in data/anomaly_detection.png")
