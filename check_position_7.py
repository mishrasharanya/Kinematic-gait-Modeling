import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ezc3d
from scipy.linalg import expm

# === Load C3D ===
c3d = ezc3d.c3d("./138_HealthyPiG_10.05/SUBJ01/SUBJ1 (0).c3d")
points = c3d['data']['points']  # shape: (4, N_markers, T)
labels = c3d['parameters']['POINT']['LABELS']['value']
T = points.shape[2]

# === Marker Tree and Parent Structure ===
marker_names = ["SACR", "RFEP", "LFEP", "RFEO", "LFEO", "RTIO", "LTIO", "RTOE", "LTOE"]
parent =       [-1,     0,      0,      1,      2,      3,      4,      5,      6]
label_lookup = {label: idx for idx, label in enumerate(labels)}

# === Helper Functions ===
def skew(w):
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

def transformation_matrix(vec, omega, q):
    v = -np.cross(omega, q)
    S = np.zeros((4, 4))
    S[:3, :3] = skew(omega)
    S[:3, 3] = vec  # Using displacement directly
    return expm(S)

# === Animation Setup ===
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(-1500, 1000)
    ax.set_title(f"Gait Animation with Screw Theory (Frame {frame})")

    # Initialize transformations list
    T_list = [np.eye(4) for _ in marker_names]
    omega = np.array([1, 0, 0])  # constant joint axis

    # Set root position from SACR marker
    root_idx = label_lookup["SACR"]
    T_list[0][:3, 3] = points[:3, root_idx, frame]

    # Apply screw-based transformation for each child
    for i in range(1, len(marker_names)):
        p_idx = parent[i]
        parent_pos = points[:3, label_lookup[marker_names[p_idx]], frame]
        child_pos = points[:3, label_lookup[marker_names[i]], frame]

        vec = child_pos - parent_pos
        q = parent_pos
        T_i = transformation_matrix(vec, omega, q)
        T_list[i] = T_list[p_idx] @ T_i

    # Extract positions after transformation
    positions = [T[:3, 3] for T in T_list]

    # Plot skeleton using parent array
    for i in range(1, len(marker_names)):
        p = parent[i]
        x = [positions[p][0], positions[i][0]]
        y = [positions[p][1], positions[i][1]]
        z = [positions[p][2], positions[i][2]]

        color = 'blue' if i in [1, 3, 5, 7] else 'red' if i in [2, 4, 6, 8] else 'gray'
        ax.plot(x, y, z, marker='o', linewidth=2, color=color)

    # Pelvis marker
    ax.scatter(*positions[0], color='black', label="SACR")
    ax.legend()

ani = FuncAnimation(fig, update, frames=min(T, 300), interval=50)
plt.show()
