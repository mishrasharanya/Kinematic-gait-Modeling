import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ezc3d

# === Load C3D file ===
c3d = ezc3d.c3d("./138_HealthyPiG_10.05/SUBJ01/SUBJ1 (0).c3d")  # <-- Replace with your actual C3D file

# === Marker labels to track ===
right_markers = ["RFEP", "RFEO", "RTIO", "RTOE"]
left_markers = ["LFEP", "LFEO", "LTIO", "LTOE"]
pelvis_marker = "SACR"
all_markers = [pelvis_marker] + right_markers + left_markers

# === Get marker data and labels ===
points = c3d['data']['points']  # shape: (4, N_markers, N_frames)
labels = c3d['parameters']['POINT']['LABELS']['value']

# === Find indices of all required markers ===
marker_indices = {}
for m in all_markers:
    if m in labels:
        marker_indices[m] = labels.index(m)
    else:
        print(f"Warning: Marker {m} not found in C3D file")

# === Extract marker positions over time ===
T = points.shape[2]
right_leg_trajectory = []
left_leg_trajectory = []
pelvis_trajectory = []

for frame in range(T):
    right_positions = []
    left_positions = []

    for m in right_markers:
        idx = marker_indices[m]
        x, y, z = points[:3, idx, frame]
        right_positions.append([x, y, z])

    for m in left_markers:
        idx = marker_indices[m]
        x, y, z = points[:3, idx, frame]
        left_positions.append([x, y, z])

    # Use SACR directly as pelvis center
    sac_idx = marker_indices[pelvis_marker]
    pelvis = points[:3, sac_idx, frame]
    pelvis_trajectory.append(pelvis)

    right_leg_trajectory.append(right_positions)
    left_leg_trajectory.append(left_positions)

# === Animate both legs + pelvis center ===
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(-1500, 1000)
    ax.set_title("C3D Gait Animation (Legs + SACR)")

    # Right leg
    right_pts = right_leg_trajectory[frame]
    xr, yr, zr = zip(*right_pts)
    ax.plot(xr, yr, zr, marker='o', color='blue', linewidth=2, label='Right Leg')

    # Left leg
    left_pts = left_leg_trajectory[frame]
    xl, yl, zl = zip(*left_pts)
    ax.plot(xl, yl, zl, marker='o', color='red', linewidth=2, label='Left Leg')

    # Pelvis center using SACR
    pelvis = pelvis_trajectory[frame]
    ax.scatter(*pelvis, marker='o', color='black', label='SACR (Pelvis Marker)')

    # Draw line from pelvis to RFEP
    rfep = right_pts[0]
    ax.plot(
        [pelvis[0], rfep[0]],
        [pelvis[1], rfep[1]],
        [pelvis[2], rfep[2]],
        color='gray', linestyle='--'
    )

    # Draw line from pelvis to LFEP
    lfep = left_pts[0]
    ax.plot(
        [pelvis[0], lfep[0]],
        [pelvis[1], lfep[1]],
        [pelvis[2], lfep[2]],
        color='gray', linestyle='--'
    )

    ax.legend()

ani = FuncAnimation(fig, update, frames=min(T, 300), interval=50)
plt.show()
