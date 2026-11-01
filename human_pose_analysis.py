
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load human pose data
file_path = "data/human_pose_data.csv"

if not os.path.exists(file_path):
    print("Human pose CSV not found!")
    print("Run human_pose_tracker.py first.")
    exit()

df = pd.read_csv(file_path)

if df.empty:
    print("No movement data available!")
    exit()

print("\nHuman Motion Analysis")
print("---------------------")

# Basic information
print("Total observations:", len(df))
print("\nBody parts detected:")
print(df["body_part"].value_counts())

# Create output directory
os.makedirs("data/pose_graphs", exist_ok=True)

# Separate graph for each body part
for part in df["body_part"].unique():

    part_data = df[df["body_part"] == part].copy()
    part_data = part_data.sort_values("time_seconds")

    # Plot trajectory
    plt.figure(figsize=(8, 6))

    plt.plot(
        part_data["x"],
        part_data["y"],
        linewidth=1.5
    )

    plt.scatter(
        part_data["x"],
        part_data["y"],
        s=8
    )

    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.title(f"{part} Movement Trajectory")

    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"data/pose_graphs/{part}_trajectory.png",
        dpi=300
    )

    plt.show()

print("\nAll body-part graphs saved!")
