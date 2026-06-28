import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

# --- Configuration ---
DATASET_PATH = "gs://gresearch/robotics/droid_100/1.0.0"
OUTPUT_DIR = "./prepared_data"
GAP_K = 5  # Frame gap (M1 primary)
TRAIN_RATIO = 0.8  # Episode-level split ratio


def extract_pairs_from_episode(episode, k=GAP_K):
    """
    Extracts (obs_t, obs_t+k, delta_q, delta_g) from a single RLDS episode.
    """
    steps = list(episode["steps"].as_numpy_iterator())

    frames_t = []
    frames_tk = []
    delta_qs = []
    delta_gs = []
    q_ts = []

    # We can only extract pairs up to len(steps) - k
    for t in range(len(steps) - k):
        step_t = steps[t]
        step_tk = steps[t + k]

        # 1. Images (Exterior View)
        # Using exterior_image_1_left as specified in the verified facts
        img_t = step_t["observation"]["exterior_image_1_left"]
        img_tk = step_tk["observation"]["exterior_image_1_left"]

        # 2. Realized Joint Displacement (Δq)
        # observation.joint_position[t+k] - observation.joint_position[t]
        q_t = step_t["observation"]["joint_position"]
        q_tk = step_tk["observation"]["joint_position"]
        delta_q = q_tk - q_t

        # 3. Gripper Change (Δg)
        # observation.gripper_position[t+k] - observation.gripper_position[t]
        g_t = step_t["observation"]["gripper_position"]
        g_tk = step_tk["observation"]["gripper_position"]
        delta_g = np.atleast_1d(g_tk - g_t)

        frames_t.append(img_t)
        frames_tk.append(img_tk)
        delta_qs.append(delta_q)
        delta_gs.append(delta_g)
        q_ts.append(q_t)

    return {
        "frames_t": np.array(frames_t),
        "frames_tk": np.array(frames_tk),
        "delta_qs": np.array(delta_qs),
        "delta_gs": np.array(delta_gs),
        "q_t": np.array(q_ts),
    }


def main():
    print(f"Loading DROID-100 builder from: {DATASET_PATH}")
    builder = tfds.builder_from_directory(builder_dir=DATASET_PATH)
    dataset = builder.as_dataset(split="train")

    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

    # Convert to list of episodes to enforce strict episode-level splitting
    print("Collecting episodes...")
    episodes = list(dataset)
    num_episodes = len(episodes)
    print(f"Found {num_episodes} episodes.")

    # Episode-level Train/Test Split
    num_train = int(num_episodes * TRAIN_RATIO)
    train_episodes = episodes[:num_train]
    test_episodes = episodes[num_train:]

    print(f"Splitting: {len(train_episodes)} Train | {len(test_episodes)} Test")

    # Process Training Data
    print("Processing Training Episodes...")
    for i, ep in enumerate(tqdm(train_episodes)):
        data = extract_pairs_from_episode(ep, k=GAP_K)
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, "train", f"ep_{i:03d}_k{GAP_K}.npz"), **data
        )

    # Process Testing Data
    print("Processing Testing Episodes...")
    for i, ep in enumerate(tqdm(test_episodes)):
        data = extract_pairs_from_episode(ep, k=GAP_K)
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, "test", f"ep_{i:03d}_k{GAP_K}.npz"), **data
        )

    print(f"Extraction complete! Data saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    # Prevent TF from preallocating all GPU memory just for data loading
    tf.config.experimental.set_visible_devices([], "GPU")
    main()
