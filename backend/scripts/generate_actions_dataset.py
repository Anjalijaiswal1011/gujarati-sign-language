"""
Generate Synthetic Dataset for dynamic action gestures (CNN+LSTM).
Creates landmark sequences of shape (30, 84) for 10 actions:
  ["walking", "eating", "drinking", "helping", "greeting",
   "running", "sitting", "standing", "jumping", "dancing"]
"""

import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQUENCES_DIR = os.path.join(BASE_DIR, "dataset", "sequences", "actions")

N_FRAMES = 30
N_LANDMARKS = 21
N_COORDS = 4  # X, Y, Z, confidence (visibility)
N_FLAT = N_LANDMARKS * N_COORDS  # 84
N_SEQUENCES = 80  # Sequences per class

ACTIONS = [
    "walking", "eating", "drinking", "helping", "greeting",
    "running", "sitting", "standing", "jumping", "dancing"
]

def get_base_pose(action_idx):
    """Generate base pose with 21 landmarks."""
    rng = np.random.default_rng(seed=action_idx * 11 + 47)
    pose = np.zeros((N_LANDMARKS, 3))
    pose[0] = [0.0, 0.0, 0.0]  # wrist
    
    # 5 fingers
    mcp_x = [-0.15, -0.07, 0.0, 0.07, 0.14]
    for i, base_lm in enumerate([1, 5, 9, 13, 17]):
        pose[base_lm] = [mcp_x[i], 0.35 + rng.uniform(-0.04, 0.04), 0.0]
        
    finger_chains = [
        (1, [2, 3, 4]),
        (5, [6, 7, 8]),
        (9, [10, 11, 12]),
        (13, [14, 15, 16]),
        (17, [18, 19, 20]),
    ]
    
    # Random extensions per action class
    ext_level = rng.uniform(0.2, 0.9, 5)
    for fi, (mcp_lm, chain) in enumerate(finger_chains):
        mcp = pose[mcp_lm]
        ext = ext_level[fi]
        seg_len = 0.10 * ext + 0.05
        for seg_i, lm in enumerate(chain):
            curve = np.sin((seg_i + 1) * 0.5 * (1 - ext)) * 0.04
            pose[lm] = mcp + np.array([0.0, seg_len * (seg_i + 1), curve])
            
    return pose

def get_motion_delta(action_idx, frame_idx, t):
    """Motion signatures for dynamic actions."""
    # Action motion patterns
    patterns = [
        np.array([0.06, 0.02, 0.01, 0.02]),  # walking
        np.array([0.02, 0.07, 0.02, 0.01]),  # eating
        np.array([0.01, 0.05, 0.06, 0.03]),  # drinking
        np.array([0.03, 0.03, 0.03, 0.04]),  # helping
        np.array([0.05, 0.04, 0.01, 0.05]),  # greeting
        np.array([0.08, 0.04, 0.02, 0.02]),  # running
        np.array([0.01, 0.01, 0.01, 0.06]),  # sitting
        np.array([0.01, 0.08, 0.01, 0.01]),  # standing
        np.array([0.02, 0.09, 0.05, 0.02]),  # jumping
        np.array([0.07, 0.06, 0.05, 0.07]),  # dancing
    ]
    
    w = patterns[action_idx % len(patterns)]
    delta = np.zeros((N_LANDMARKS, 3))
    
    # Apply time-varying waveforms
    delta[:, 0] += w[0] * np.sin(2 * np.pi * t)
    delta[:, 1] += w[1] * np.sin(2 * np.pi * t * 1.5)
    delta[:, 2] += w[2] * np.cos(2 * np.pi * t)
    delta[5:17, 0] += w[3] * np.sin(np.pi * t) * 0.3
    
    return delta

def normalize_pose(pose):
    """Normalize centered at wrist."""
    wrist = pose[0]
    centered = pose - wrist
    mcp = centered[9]
    dist = np.sqrt(np.sum(mcp**2))
    if dist > 0:
        return centered / dist
    return centered

def generate_sequences():
    print("=" * 60)
    print("Generating dynamic action sequences dataset...")
    print(f"Output path: {SEQUENCES_DIR}")
    print("=" * 60)
    
    os.makedirs(SEQUENCES_DIR, exist_ok=True)
    
    for action_idx, action in enumerate(ACTIONS):
        action_dir = os.path.join(SEQUENCES_DIR, action)
        os.makedirs(action_dir, exist_ok=True)
        
        for seq_i in range(N_SEQUENCES):
            filename = f"syn_action_{seq_i:04d}.npy"
            filepath = os.path.join(action_dir, filename)
            
            # Generate 30 frames
            rng = np.random.default_rng(seed=action_idx * 500 + seq_i)
            base = get_base_pose(action_idx)
            
            sequence_84 = np.zeros((N_FRAMES, N_FLAT), dtype=np.float32)
            
            for f in range(N_FRAMES):
                t = f / (N_FRAMES - 1)
                pose = base.copy()
                pose += get_motion_delta(action_idx, f, t)
                pose += rng.normal(0, 0.008, pose.shape)  # noise
                norm_pose = normalize_pose(pose)
                
                # Hand landmarks (21, 4) - x, y, z, confidence (1.0)
                frame_84 = np.zeros((N_LANDMARKS, 4))
                frame_84[:, :3] = norm_pose
                frame_84[:, 3] = 1.0  # confidence
                
                sequence_84[f] = frame_84.flatten()
                
            np.save(filepath, sequence_84)
            
        print(f"[OK] {action:<12s} ({N_SEQUENCES} sequences generated)")
        
    print("\nAction dataset generation complete!\n")

if __name__ == "__main__":
    generate_sequences()
