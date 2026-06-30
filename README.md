# IDM toy project

A small **Inverse Dynamics Model (IDM)** trained on the real DROID-100 Franka
dataset. Given two camera frames `(obs_t, obs_{t+k})`, it tries to predict the realized joint
displacement `Δq ∈ R⁷` and gripper change `Δg ∈ R¹` — what the arm actually did
between the two frames, computed as `joint_position[t+k] − joint_position[t]` (not from
the stored velocity command).

The aim is a prerequisite gate: a vision IDM should beat a proprioception-only model
and the naive baselines on held-out episodes. If it does, the model is using the image
rather than just the current pose. This report covers the model, the baselines, and
what the measurements showed.

## What is implemented

- **Data extraction** (`dataset/main.py`)
- **Vision IDM** (`project/model.py`, `project/train.py`)
- **Proprioception-only baseline** (`project/train_proprio_baseline.py`)
- **Zero-motion & mean-Δq baselines + diagnostics** (`project/diagnostics.py`)

## Layout

```
geoidm-probe/
  Dockerfile                       # PyTorch + CUDA + uv image for a GPU box
  dataset/
    main.py                        # DROID-100 (gs://gresearch/robotics/droid_100) -> .npz pairs
  project/
    model.py                       # VisionIDM: shared ResNet18 encoder -> MLP head
    train.py                       # vision IDM trainer (two-phase), saves idm_vision.pt
    train_proprio_baseline.py      # MLP from q_t only (the baseline to beat)
    diagnostics.py                 # zero/mean baselines + the two diagnostics below
```

## The model

`VisionIDM` runs a shared ResNet18 (ImageNet weights) over both frames, applies a
1×1 conv bottleneck (512→64), flattens (64·6·10 = 3840 per frame), concatenates the two
feature vectors, and feeds a 2-layer MLP head producing 8 outputs (7 joints + 1
gripper). It can optionally concatenate `q_t` for a proprioception ablation. Training
runs in two phases: the encoder is frozen while only the head trains, then ResNet
`layer3`/`layer4` are unfrozen at a lower learning rate. The loss is L1, matching the
MAE metric.

## Data

DROID-100 is the official 2 GB, 100-trajectory RLDS slice. We use the
`exterior_image_1_left` view (native 180×320×3 uint8), `GAP_K = 5` (DROID runs at
15 Hz, so ~0.33 s between frames), and an episode-level 80/20 split. The split is by
episode rather than by pair because adjacent frames are nearly identical, and a random
pair split would leak the test set.

- ~23k train pairs, ~8.5k test pairs.
- The targets are concentrated near zero with a long tail: 46% of joint deltas have
  `|Δq| < 0.01` rad, the median joint moves about 0.012 rad, p90 is 0.06–0.13 rad, and
  the max reaches ~0.4 rad. Per-joint std is 0.04–0.08 rad.

## Running

Everything runs with `uv` (two separate uv projects, `dataset/` and `project/`). The
`Dockerfile` builds a PyTorch 2.x + CUDA 12.8 + uv image for a GPU box (an RTX 3090 was
used).

```bash
# Build / run the container, mount the repo at /workspace
docker build -t pytorch-image .
docker run --name pytorch-container --device nvidia.com/gpu=all --ipc=host -it -v "$PWD":/workspace pytorch-image

# 1. Extract pairs (isolated TF/tfds env) -> dataset/prepared_data/{train,test}/*.npz
cd /workspace/dataset && uv run main.py

# 2. Train the vision IDM -> saves idm_vision.pt
cd /workspace/project && uv run train.py

# 3. Train the proprio-only baseline
uv run train_proprio_baseline.py

# 4. Diagnostics (data + zero/mean baselines + inline proprio MLP)
uv run diagnostics.py
uv run diagnostics.py --checkpoint idm_vision.pt   # also diagnose the trained model
```

Data extraction is kept in its own TF/tfds env and run once, so training never imports
TensorFlow.

## Results

On held-out episodes, joint MAE (radians) and gripper MAE:

| predictor                | joint MAE  | gripper MAE |
| ------------------------ | :--------: | :---------: |
| zero-motion (`Δq=0`)     |   0.0380   |   0.0484    |
| mean-Δq                  |   ~0.038   |   ~0.048    |
| proprio-only MLP (`q_t`) |   0.0387   |   0.0494    |
| **vision IDM**           | **0.0384** | **0.0485**  |

All four reach about the same MAE, so the vision IDM does not clear the gate. The MAE
metric is saturated here: under L1 the best constant prediction is the per-dimension
median (≈0), and because so many targets are near zero, any model scores well simply by
predicting ≈0. To check whether the tie hides a real signal, `diagnostics.py` runs two
tests.

## Diagnostics

The two diagnostics separate two explanations for the tie:

- **World A — there is signal, but the metric hides it.** The Δq targets are small and
  roughly zero-mean, so L1/L2 is minimized at the conditional mean. A model with real
  visual signal could still look flat on MAE while its predictions correlate with the
  truth. The fix is to change the metric, not the data.
- **World B — the moving pairs are too few to dominate training.** A real tail of
  moving pairs exists, but most pairs are near-stationary, so the loss is dominated by
  them and the model can win by predicting ≈0. The fix is to filter the stationary
  pairs (and standardize the targets) so the moving pairs drive training.

**Diagnostic 1 — is there recoverable signal (correlation / sign / R²)?** Correlation
is independent of scale: a constant predictor has ≈0 correlation however it is scaled,
so any consistent positive per-joint correlation would mean there is signal to recover.
On the test split the vision IDM's mean per-joint `corr(Δq_pred, Δq_true)` is **0.009**
with R² ≤ 0. Its predictions are essentially uncorrelated with the truth, and its mean
`|Δq_pred|` ≈ 0.0038 matches the mean baseline rather than the truth's 0.0380, so it
has converged to predicting the dataset mean. A simple MLP on `q_t` alone reaches a
higher correlation (0.074) than the full ResNet on images.

**Diagnostic 2 — how much motion is there to see (`|Δq|` histogram)?** 46% of joint
entries are below 0.01 rad and the median joint moves only ~0.012 rad, so most pairs
are near-stationary and the loss is dominated by them. A real tail does exist (per-joint
p90 ≈ 0.06–0.13 rad, max ≈ 0.4), so there is learnable motion in the data even though
it is a minority of the pairs.

### Reading

Together these point mostly to World B. The correlation test shows the IDM recovered
little visual signal, rather than a signal that MAE merely hid, which is consistent with
training being driven by the near-stationary pairs instead of the moving ones. The one
joint where vision reaches corr ≈ 0.09 (j1) is the same joint where the constant mean
baseline already gets ~60% sign agreement from a consistent drift, and vision's R²
there is negative — so this is bias, not vision. The proprio MLP's higher correlation
fits the copycat / causal-confusion effect: smooth teleoperation makes the next move
partly predictable from the current pose.

These conclusions apply to this setup (k=5, one downsampled view, L1 loss) and not to
IDMs in general, and the ~100-episode slice limits how far any single number can be
pushed.

## Next steps

The tail in the data suggests there is room to improve once the moving pairs carry more
of the training signal. Ideas for improvement:

1. **Filter the stationary pairs** (train and evaluate on `Σ|Δq| > threshold`) and/or
   **standardize the targets per joint**, so the moving pairs drive training. The motion
   is already in the data; this just stops the near-zero pairs from dominating the loss.
2. **Report skill = 1 − MAE/MAE_zero and the correlation table** rather than raw MAE.
   Correlation rising above the proprio MLP's 0.074 is the clearer test that vision is
   contributing.
3. **Increase `GAP_K`** (re-extract at k=15–30), if filtering is not enough, to enlarge
   the motion within each pair.
