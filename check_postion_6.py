import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import ezc3d
import pandas as pd

# === Load C3D ===
c3d = ezc3d.c3d("./138_HealthyPiG_10.05/SUBJ01/SUBJ1 (0).c3d")
points = c3d['data']['points']
labels = c3d['parameters']['POINT']['LABELS']['value']
T = points.shape[2]

# === Marker Tree and Parent Structure ===
marker_names = ["SACR", "RFEP", "LFEP", "RFEO", "LFEO", "RTIO", "LTIO", "RTOE", "LTOE"]
parent =       [-1,     0,      0,      1,      2,      3,      4,      5,      6]
label_lookup = {label: idx for idx, label in enumerate(labels)}

# === Helper Functions ===
def translation_matrix(from_pt, to_pt):
    T = np.eye(4)
    T[:3, 3] = to_pt - from_pt
    return T

def angle_between(v1, v2):
    """Compute angle (in degrees) between two vectors using dot product."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(v1 / norm1, v2 / norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

# === Plot Setup ===
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
angle_records = []

# === Frame Update ===
def update(frame):
    ax.clear()
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(-1500, 1000)
    ax.set_title(f"Polar Coordinate System (Frame {frame})")

    T_list = [np.eye(4) for _ in marker_names]

    # Root: SACR translation
    sacr_idx = label_lookup["SACR"]
    T_list[0][:3, 3] = points[:3, sacr_idx, frame]

    for i in range(1, len(marker_names)):
        p = parent[i]
        parent_name = marker_names[p]
        child_name = marker_names[i]

        parent_pos = points[:3, label_lookup[parent_name], frame]
        child_pos = points[:3, label_lookup[child_name], frame]

        T_list[i] = translation_matrix(parent_pos, child_pos) @ T_list[p]

    # === Plot skeleton ===
    positions = [T[:3, 3] for T in T_list]
    for i in range(1, len(marker_names)):
        p = parent[i]
        x = [positions[p][0], positions[i][0]]
        y = [positions[p][1], positions[i][1]]
        z = [positions[p][2], positions[i][2]]
        color = 'blue' if i in [1, 3, 5, 7] else 'red' if i in [2, 4, 6, 8] else 'gray'
        ax.plot(x, y, z, marker='o', linewidth=2, color=color)

    ax.scatter(*positions[0], color='black', label="SACR (Root)")
    ax.plot([], [], color='blue', label='Right Leg')
    ax.plot([], [], color='red', label='Left Leg')
    ax.legend()

    # === Compute Joint Angles ===
    def get_pos(label): return points[:3, label_lookup[label], frame]
    try:
        frame_data = {'Frame': frame}

        # Segment vectors
        rhip = get_pos("RFEP"); rknee = get_pos("RFEO"); rankle = get_pos("RTIO"); rtoe = get_pos("RTOE")
        lhip = get_pos("LFEP"); lknee = get_pos("LFEO"); lankle = get_pos("LTIO"); ltoe = get_pos("LTOE")
        sacr = get_pos("SACR")

        # Vectors
        pelvic_r = rhip - sacr
        pelvic_l = lhip - sacr
        thigh_r = rknee - rhip
        thigh_l = lknee - lhip
        shank_r = rankle - rknee
        shank_l = lankle - lknee
        foot_r = rtoe - rankle
        foot_l = ltoe - lankle

        # Joint angles using angle between vectors
        hip_r = angle_between(pelvic_r, thigh_r)
        hip_l = angle_between(pelvic_l, thigh_l)
        knee_r = angle_between(thigh_r, shank_r)
        knee_l = angle_between(thigh_l, shank_l)
        ankle_r = angle_between(shank_r, foot_r)
        ankle_l = angle_between(shank_l, foot_l)

        # Averaged joint angles (with 90 offset for hip/ankle)
        frame_data['HipAngle'] = 90 - (hip_r + hip_l) / 2
        frame_data['KneeAngle'] = (knee_r + knee_l) / 2
        frame_data['AnkleAngle'] = 90 - (ankle_r + ankle_l) / 2

        if frame == 0:
            angle_records.clear()
        angle_records.append(frame_data)
    except Exception as e:
        print(f"Error at frame {frame}: {e}")

# === Create Animation ===
ani = FuncAnimation(fig, update, frames=min(T, 300), interval=50)

# === Show Animation Window (Optional) ===
plt.show()

# === Save Joint Angles to CSV ===
angle_df = pd.DataFrame(angle_records)
angle_df = angle_df[['Frame', 'HipAngle', 'KneeAngle', 'AnkleAngle']]
angle_df.to_csv("joint_angles_spherical.csv", index=False)
print("Joint angle data saved to joint_angles_corrected.csv")

# === Save Animation as GIF ===
#ani.save("joint_angle_animation.gif", writer=PillowWriter(fps=20))
#print("GIF saved as 'joint_angle_animation.gif'")
