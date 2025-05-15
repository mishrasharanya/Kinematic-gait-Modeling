import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import ezc3d
from scipy.linalg import expm
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
def skew_symmetric(w):
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

def compute_theta(vec, omega):
    if np.linalg.norm(vec) < 1e-6:
        return 0
    vec_norm = vec / np.linalg.norm(vec)
    return np.arccos(np.clip(np.dot(vec_norm, omega), -1.0, 1.0))


def translation_matrix(from_pt, to_pt):
    T = np.eye(4)
    T[:3, 3] = to_pt - from_pt
    return T

def unit_vec(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else np.zeros_like(v)

def signed_angle(v1, v2, normal_axis):
    u1 = unit_vec(v1)
    u2 = unit_vec(v2)
    angle = np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return np.degrees(angle)

# === Fixed ω for Joints (simplified anatomical assumptions) ===
omega_lookup = {
    "RFEP": np.array([1, 0, 0]),
    "LFEP": np.array([1, 0, 0]),
    "RFEO": np.array([1, 0, 0]),
    "LFEO": np.array([1, 0, 0]),
    "RTIO": np.array([1, 0, 0]),
    "LTIO": np.array([1, 0, 0]),
    "RTOE": np.array([1, 0, 0]),
    "LTOE": np.array([1, 0, 0]),
}

# === Plot Setup ===
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
theta_buffer = np.zeros((len(marker_names), T))  # θ across frames
angle_records = []

# === Frame Update ===
def update(frame):
    ax.clear()
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(-1500, 1000)
    ax.set_title(f"Cartesian Cordinate System (Frame {frame})")

    T_list = [np.eye(4) for _ in marker_names]

    # === Root: SACR known ===
    sacr_idx = label_lookup["SACR"]
    T_list[0][:3, 3] = points[:3, sacr_idx, frame]

    # === Compute other poses & spatial velocities ===
    for i in range(1, len(marker_names)):
        child = marker_names[i]
        p_idx = parent[i]
        parent_name = marker_names[p_idx]

        parent_pos = points[:3, label_lookup[parent_name], frame]
        child_pos = points[:3, label_lookup[child], frame]

        iTp = translation_matrix(parent_pos, child_pos)
        T_list[i] = iTp @ T_list[p_idx]

        omega = omega_lookup.get(child, np.array([0, 0, 1]))
        vec = child_pos - parent_pos
        theta = compute_theta(vec, omega)
        theta_buffer[i, frame] = theta
        dtheta = theta_buffer[i, frame] - theta_buffer[i, frame - 1] if frame > 0 else 0.0


    # === Plot skeleton & velocity arrows ===
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

    # === Compute Signed Joint Angles (without Pelvis) ===
    def get_pos(label): return points[:3, label_lookup[label], frame]
    try:
        frame_data = {'Frame': frame}

        # Right
        rhip = get_pos("RFEP"); rknee = get_pos("RFEO"); rankle = get_pos("RTIO"); rtoe = get_pos("RTOE")
        thigh_r = rknee - rhip; shank_r = rankle - rknee; foot_r = rtoe - rankle
        right_hip_angle = signed_angle(rhip - get_pos("SACR"), thigh_r, np.array([1, 0, 0]))
        right_knee_angle = signed_angle(thigh_r, shank_r, np.array([1, 0, 0]))
        right_ankle_angle = signed_angle(shank_r, foot_r, np.array([1, 0, 0]))

        # Left
        lhip = get_pos("LFEP"); lknee = get_pos("LFEO"); lankle = get_pos("LTIO"); ltoe = get_pos("LTOE")
        thigh_l = lknee - lhip; shank_l = lankle - lknee; foot_l = ltoe - lankle
        left_hip_angle = signed_angle(lhip - get_pos("SACR"), thigh_l, np.array([1, 0, 0]))
        left_knee_angle = signed_angle(thigh_l, shank_l, np.array([1, 0, 0]))
        left_ankle_angle = signed_angle(shank_l, foot_l, np.array([1, 0, 0]))

        # === Apply Offsets for Plug-in Gait Style Angles ===
        frame_data['HipAngle'] = 90 - (left_hip_angle + right_hip_angle) / 2
        frame_data['KneeAngle'] = (left_knee_angle + right_knee_angle) / 2
        frame_data['AnkleAngle'] = 90 - (left_ankle_angle + right_ankle_angle) / 2

        if frame == 0:
            angle_records.clear()
        angle_records.append(frame_data)
    except:
        pass

# === Run Animation ===
ani = FuncAnimation(fig, update, frames=min(T, 300), interval=50)
plt.show()

# === Save angles to CSV (no PelvisAngle) ===
angle_df = pd.DataFrame(angle_records)
angle_df = angle_df[['Frame', 'HipAngle', 'KneeAngle', 'AnkleAngle']]
angle_df.to_csv("joint_angles_output.csv", index=False)
print("Joint angle data saved to joint_angles_output.csv")

# === Save Animation as GIF ===
ani.save("gait_animation.gif", writer=PillowWriter(fps=20))
print("Animation saved as 'gait_animation.gif'")