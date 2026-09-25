# Replication Notes

Working notes for replicating *Recurrent Model-Free RL Can Be a Strong Baseline for
Many POMDPs* (Ni, Eysenbach & Salakhutdinov, ICML 2022) on macOS / Apple Silicon.

Everything here was established by running the code, not by reading about it. Numbers
are labelled **measured** or **modelled**; don't cite a modelled number without saying so.

**Status:** environment reproduced and training verified end to end. No completed
training run yet. Resume support not yet implemented.

---

## 1. Quickstart

```bash
# build the faithful replication image (amd64, exact paper-era pins)
docker build --platform linux/amd64 -t pomdp-repl .

# smallest cross-validatable run: Pendulum-V, ~50k env steps
docker run --rm -it -v "$PWD":/workspace pomdp-repl \
  python policies/main.py --cfg configs/pomdp/pendulum/v/rnn.yml --algo sac --cuda -1

# cheapest run of all: CartPole-V, ~11k env steps
docker run --rm -it -v "$PWD":/workspace pomdp-repl \
  python policies/main.py --cfg configs/pomdp/cartpole/v/rnn.yml --cuda -1
```

`--cuda -1` forces CPU. `--debug` writes to `debug/` instead of `logs/`.

Two images exist, and the distinction matters:

| image | stack | use for |
|---|---|---|
| `Dockerfile` (amd64) | torch **1.7.1**, numpy 1.20.1 — the paper's versions | **anything you intend to report** |
| `Dockerfile.arm64` | torch **1.13.1**, numpy 1.23.5 — drifted | speed experiments only, never for results |

The arm64 lane exists because no arm64 build of torch 1.7.1 has ever existed. It runs
~2x faster natively but produces different trajectories, so **never pool results across
the two lanes**.

---

## 2. Why the documented setup does not work

`conda env create -f environments.yml` fails on Apple Silicon before installing anything.
The file is a `conda env export` from the authors' Linux cluster and contains packages
with no macOS build: `cudatoolkit=11.0.221`, `gcc_linux-64=7.3.0`, `binutils_linux-64`,
`_sysroot_linux-64_curr_repodata_hack`, `mujoco-py==2.1.2.14`, and `pytorch=1.7.1`
(no `osx-arm64` build; Apple Silicon support landed in torch 1.12).

Do not try to repair that file. Build a fresh environment instead — that is what
`Dockerfile` is.

### Hard version ceilings in the source

These are not preferences. Each one is a crash.

| ceiling | where | symptom above it |
|---|---|---|
| **Python <= 3.9** | `utils/logger.py:9` — `from collections import OrderedDict, Set` | `collections.Set` removed in 3.10 → `ImportError` on first import |
| **numpy < 1.24** | `torchkit/pytorch_utils.py:73` — `v.dtype == np.bool` | `np.bool` alias removed in 1.24 → `AttributeError` |
| **gym 0.21.x** | 4-tuple `step()` in `envs/pomdp/wrappers.py:43`, `utils/helpers.py:50`, `policies/learner.py:466,624`, `utils/evaluation.py:145`; `env.seed()` in `policies/learner.py:109-110`, `utils/system.py:12-13` | gym >= 0.26 returns a 5-tuple → `ValueError: too many values to unpack`; `env.seed()` removed |
| **`future-fstrings` installed** | coding line in `policies/main.py`, `policies/learner.py`, `torchkit/pytorch_utils.py` | `SyntaxError: unknown encoding: future_fstrings` |

The seeding calls are load-bearing (`learner.py:110` is even commented `# crucial`), so a
partial gymnasium port would silently change your numbers rather than crash. Don't.

---

## 3. Dependency ledger

Every pin in `Dockerfile`, and the specific reason. Seven of these are **not** mentioned
anywhere in the README — they were found by hitting them.

| pin | why |
|---|---|
| `python:3.8-slim-bullseye` | 3.9 is the ceiling (see above); 3.8 matches `environments.yml` |
| `pip==22.0.4 setuptools==65.5.0 wheel==0.38.4` | gym 0.21's `setup.py` fails under modern setuptools (`'extras_require' must be a dictionary`). Must be pinned **before** installing gym. |
| `torch==1.7.1+cpu` | the paper's version; CPU build because no CUDA here |
| `numpy==1.20.1` | matches `environments.yml`; also below the `np.bool` cliff |
| `gym==0.21.0` | legacy API (see above). sdist-only, builds under the pinned setuptools. |
| `pillow==7.2.0` | **undocumented.** Newer Pillow imports `numpy.typing.NDArray`, absent in numpy 1.20 → breaks matplotlib/seaborn import chain |
| `protobuf==3.19.4` | **undocumented.** tensorboard 2.8's generated `_pb2` files reject protobuf >= 4 (`Descriptors cannot be created directly`) |
| `tensorboardx==1.8` | **undocumented.** `utils/logger.py:14` falls back to it when the torch tensorboard import fails |
| `scikit-learn==0.23.2`, `scipy==1.6.0`, `joblib==1.0.1` | **undocumented.** `utils/evaluation.py:13` imports `sklearn.manifold.TSNE` unconditionally |
| `box2d==2.3.10` | **undocumented.** `envs/pomdp/__init__.py:59` constructs `LunarLander-v2` **at import time**, so Box2D is required even for a Pendulum-only run |
| `pycolab==1.2` | **undocumented.** `policies/learner.py:105` imports `envs.credit_assign` unconditionally on the `pomdp` branch, which pulls in Key-to-Door |
| `pybullet==3.1.8` | **drift.** `environments.yml` says 3.1.0, which is sdist-only (PyPI wheels jump 3.0.9 → 3.1.8). Inert for Pendulum/CartPole — `envs/pomdp/__init__.py:79` only imports it to register the BLT envs. **Revisit before running any `*BLT-*` config.** |

### Build gotchas

- **No `apt` layer.** Debian buster *and* bullseye archive pools now return 404, so
  `apt-get install` fails. Everything above resolves to a prebuilt wheel, so no compiler
  is needed and `apt` is avoided entirely.
- **arm64 has no wheels** for `pybullet` or `box2d` at any version (box2d sdists stop at
  2.3.2), which is why `Dockerfile.arm64` uses a non-slim base that ships gcc.
- **torch 1.13.1 needs numpy >= 1.22** (compiled against C API 0xf; numpy 1.20 is 0xe →
  `RuntimeError: Numpy is not available`). Combined with the `np.bool` cliff at 1.24,
  the arm64 lane has exactly one usable window: **numpy 1.23.5**.

---

## 4. Known staleness in the repo

- **`example.ipynb` fails as written.** The README calls it the "minimal example", but it
  predates the camera-ready refactor. Cell 4 passes `state_embedding_size=` and `algo=`;
  `policies/models/policy_rnn.py:32` now expects `observ_embedding_size=` and
  `algo_name=`, and `algo_name` requires a nested per-algorithm kwarg dict, e.g.
  `td3=dict(exploration_noise=0.1, target_noise=0.2, target_noise_clip=0.5)`.
  **Use `policies/main.py` instead.**
- **`docs/our_details.md` links `buffers/seq_replay_buffer.py`**, which was split into
  `_vanilla` and `_efficient`.
- **Jupyter, unrelated to this repo:** Homebrew `jupyterlab` 4.6.3 ships tornado 6.5.9
  with jupyter_server 2.21.0, and every static asset 500s with
  `AttributeError: 'FileFindHandler' object has no attribute 'allowed_symlink_directory'`.
  Tornado >= 6.5.2 added `_resolve_symlink_target`, which jupyter_server's `FileFindHandler`
  subclass never initialises. Pin `tornado<6.5.2`, or just don't use Jupyter here.

---

## 5. What the sanity-check run actually is

`configs/pomdp/pendulum/v/rnn.yml` **as shipped** is a variant the paper published, so it
needs no tuning. From `scripts/eval.sh:5-10`:

```bash
python scripts/plot_diagnose.py --csv_path results/data/pomdp/Pendulum/V/final.csv \
    --max_x 50000 --window_size 3 --instances sac-lstm-200-oar-separate,sac-lstm-200-oar-shared
```

`sac-lstm-200-oar-separate` decodes onto the config field by field:

| tag | config |
|---|---|
| `sac` | `policy.algo_name: sac` |
| `lstm` | `policy.seq_model: lstm` |
| `200` | `train.sampled_seq_len: -1` → resolves to `max_trajectory_len` = 200 |
| `oar` | all three embedding sizes non-zero (32 / 8 / 8) |
| `separate` | `policy.separate: True` |

`--max_x 50000` matches 250 iters x 200 steps. **Cross-validation target:** download
`data.zip` (linked from `docs/plot_curves.md`) into `results/`, then compare your
`metrics/return_eval_total` against the `sac-lstm-200-oar-separate` column of
`results/data/pomdp/Pendulum/V/final.csv`. Compare distributions over >= 3 seeds, never
single curves.

### What one run outputs

```
logs/pomdp/Pendulum/V/sac_lstm/gamma-0.9/len--1/bs-32/freq-1.0/oar/<timestamp>/
├── progress.csv            # the deliverable: one row per eval = the learning curve
├── experiment.log          # stdout: arch printout, per-episode returns
├── events.out.tfevents.*   # tensorboard, same metrics
├── variant_<pid>.yml       # resolved config including the seed
├── policies/               # verbatim source snapshot at launch
└── save/agent_<iter>_perf<x>.pt   # weights only, and only past 75% of training
```

Key columns: `z/env_steps` (x-axis), `metrics/return_eval_total` (the headline),
`rl_loss/*` (diagnostics), `z/fps` and `z/time_cost` (throughput).

**The seed is deliberately not in the log path** — everything in the path is a
hyperparameter. `scripts/merge_csv.py:25` globs `**/progress.csv` under a config prefix
and treats identical prefixes as repeats to average, keying them by timestamp. So just
launch with different `--seed` and the plotting pipeline finds them. Corollary: **never
hand-edit those directory names, the path is the metadata.**

---

## 6. Compute budget

### The three counters

| unit | what it is | cost |
|---|---|---|
| **env step** | one `env.step()`, one stored transition | 448 us — **23 s of the entire run** |
| **episode** | `reset()` to time limit; length from `envs/pomdp/__init__.py` | sets sequence length |
| **update** | one `agent.update(batch)`; never touches the env | **1.58 s — effectively 100% of runtime** |

`policies/learner.py:376-387` is the whole loop: collect `num_rollouts_per_iter`
episodes, then do `num_updates_per_iter` x env_steps updates.

**`num_updates_per_iter` being a float means it is a ratio**, not a count
(`learner.py:init_train`). `1.0` = one gradient update per env step collected. That single
line is why Pendulum-V needs 51,000 updates for 255 episodes of experience.

### Where one update goes (measured, emulated amd64, 1 thread)

| phase | ms | share |
|---|---|---|
| `critic_loss` — 3 RNN forwards (actor, critic_target, critic) | 484 | 30.7% |
| critic backward — BPTT through 201 steps | 255 | 16.2% |
| `actor_loss` — 2 RNN forwards (actor, critic) | 325 | 20.6% |
| actor backward | 469 | 29.7% |
| both `optimizer.step()` + `soft_target_update` | 9 | 0.5% |

Forwards 51%, backwards 46%, bookkeeping 0.5%. **No hidden overhead — it is all
sequence.** Replay-buffer sampling is 0.3–0.8%.

### Measured per-update costs

Emulated amd64 (this laptop), torch 1.7.1, 1 thread, via a synthetic-batch harness:

| config | shape | ms/update |
|---|---|---|
| Pendulum-V | T=201 B=32 obs=1 | 1794.5 |
| CartPole-V | T=201 B=32 obs=2 (`sacd`) | 1668.2 |
| `*BLT-*` | T=64 B=64 obs=29 | 1467.7 |
| Semi-Circle | T=120 B=32 obs=2 | 849.4 |
| Cheetah-Vel | T=400 B=32 obs=21 | 4091.7 |
| Pendulum, notebook shape | T=64 B=32 obs=1 | 549.6 |
| **Markovian** (`mlp.yml`) | batch 64, no sequence | **19.6** |

**Threads do not help much.** Measured: 1 thread 1628 ms, 2 threads 1127 ms (1.44x),
4 threads 1.85x, 8 threads collapses. `policies/main.py:66` hardcodes
`torch.set_num_threads(1)` and that is already near-optimal — the LSTM's 201 timesteps are
strictly serial, so there is little to parallelise inside one update.

### The cost of memory

Same environment, same 1,505 episodes, same 1.5M updates — only the batch shape changes:

| policy | rows/update | ms/update | h/seed |
|---|---|---|---|
| Markovian (`mlp`, batch 64) | 64 | 19.6 | **8.2** |
| Recurrent (`lstm`, 64x64 window) | 4,096 | 1467.7 | **612** |

**75x.** Which is the paper's thesis from the other side: "recurrent model-free RL is a
strong baseline" is a claim about *sample* efficiency. It says nothing about compute, and
the compute is a 75x tax. Every baseline in the paper is nearly free; the recurrent arm is
the entire budget.

### Per-run durations

Laptop figures are measured x read-from-config. GPU figures are **modelled**.

| run | updates | laptop | native x86 | 1x A100 (modelled) |
|---|---|---|---|---|
| CartPole-V | 11,000 | 5.1 h | 2.4–3.6 h | 8–15 min |
| Pendulum-V | 51,000 | 25.4 h | 12–17 h | 38 min – 1.3 h |
| Cheetah-Vel | 100,000 | 114 h | ~57 h | 2.8–5.7 h |
| Ant-Dir | 600,000 | 682 h | ~341 h | 17–34 h |
| **AntBLT-P** | 1,500,000 | **612 h** | ~306 h | **15–31 h** |

### Whole-repo total

52 config files x 5 seeds = 260 runs:

```
WORK   = 249,342,600 gradient updates = 884 PFLOP (0.88 EFLOP)
LAPTOP = 53,351 hours = 2,223 days = 6.1 YEARS serial
```

The Markovian arm is **52% of all updates but 1.2% of all time.**

### Scheduling: what parallelism is available

```
grid: envs x arms x variants x seeds   <- embarrassingly parallel   YES
  ────────────────────────────────────────────────────────────────
  one run (config, seed)               <- THE BOUNDARY
  ────────────────────────────────────────────────────────────────
    ~50k-1.5M updates                  <- strictly serial           NO
      201 timesteps per update         <- strictly serial           NO
```

A run is the smallest unit you can put on a different machine. It is **indivisible in
space** but *divisible in time* once resume exists.

```
makespan(N) = max( total_work / N ,  longest_single_run )
```

Job sizes are wildly unequal — median run 0.80 GPU-h, longest 28.4 GPU-h, and **half the
runs are 1.4% of the work**. So queueing is free; only the critical path costs you.
Scheduled with longest-processing-time-first:

| slots | makespan | utilization |
|---|---|---|
| 12 | 111 h | 100% |
| 24 | 56 h | 100% |
| **48** | **28.4 h** | 98% ← floor reached |
| 100 | 28.4 h | 47% |
| 260 | 28.4 h | 18% |

**48 slots is the useful ceiling; 260 is the absolute one.** Past 48, extra workers finish
early and idle waiting for the single longest run.

The floor is set by one config, and cascades if you drop it:

| scope | longest job | floor |
|---|---|---|
| everything | `credit/catch/rnn` | 28.4 h |
| drop `credit/` | `meta/humanoid_dir/rnn` | 24.9 h |
| **pomdp only** | any `*_blt/*/rnn` | **12.6 h** |

**Planning rule: your most expensive single config sets the turnaround time for the whole
grid, regardless of hardware.** Choose it deliberately.

### GPUs

**There is no multi-GPU support.** `grep -rE "DataParallel|DistributedDataParallel|nccl|device_ids"` over
`policies/ torchkit/ buffers/ utils/` returns nothing, and `torchkit/pytorch_utils.py:106`
is a single device:

```python
device = torch.device(f"cuda:{gpu_id}" if _use_gpu else "cpu")
```

Even if added, data parallelism would hurt: batch 32 split 4 ways is batch 8 per device,
and you would all-reduce **2.28 MB** of gradients (597,288 parameters) 50,000 times per run.

**The GPU win is kernel fusion, not parallelism.** cuDNN runs all 201 timesteps in one
launch instead of the CPU's hundreds of dispatches. Measured footprint says the card is
otherwise idle: 10.4 GFLOP per update is **0.53 ms** of arithmetic on an A100's ~19.5
TFLOP/s, against a realistic ~40 ms per update — roughly **1.3% utilization**.

That idleness is not bad news. It is why **8–16 independent runs fit on one card**, which
is where the real multiplier lives:

```
laptop  ->  1 A100            ~ 40x   (kernel fusion)
        ->  + 12 packed runs  ~ 12x   (the card was idle anyway)
        =  ~480x on a single card
        ->  4 cards            ~3.9x more, then floor-bound
```

**Speedup estimate, and how it was derived.** The README claims the notebook costs
"< 20 min" on a GPU. That config is 30,000 updates at `sampled_seq_len=64`, whose exact
shape measures **549.6 ms/update** here → implies <= 40 ms/update on the authors'
(2021-era, V100-class) GPU → **>= 13.7x**. An A100 adds maybe 1.5–2.5x on a
latency-bound workload. So:

> **laptop -> one A100: roughly 20–40x.**

This is the weakest number in this document. It rests on a prose claim in a README, not a
measurement. **To settle it:** run the per-update harness on any rented CUDA box with
`ptu.set_gpu_mode(True)` — ten minutes replaces the whole range with a fact.

---

## 7. Runners

### Validity rule

Three surfaces, two of them identical:

| surface | stack | device |
|---|---|---|
| laptop (Docker) | torch 1.7.1 | amd64 CPU, emulated |
| GitHub Actions | torch 1.7.1 | amd64 CPU, **native — bitwise-identical to laptop** |
| Kaggle | torch 1.7.1+cu110 | **GPU** |

Same seed + same stack = identical trajectory (`utils/system.py:reproduce()` seeds numpy,
random and torch; `learner.py:109-110` seeds the env and action space). But **CPU and GPU
produce different trajectories** for the same seed — different RNG streams and kernels.

> **Run each complete comparison on one surface.** All arms (`rnn` / `mlp` / `oracle`),
> all seeds, same place. Never split one comparison across CPU and GPU.

Different seeds of the same config on different *machines with the same stack* are fine —
pool them freely. That is what the container is for.

### GitHub Actions

Free and unlimited for **public** repositories, **20 concurrent jobs** (Free plan),
**6 h hard cap per job**. Runners are native x86-64 with Docker preinstalled, so
`Dockerfile` runs unchanged. See `.github/workflows/train.yml` (manual dispatch only —
nothing fires on push).

| run | fits one job? |
|---|---|
| CartPole-V (2.4–3.6 h) | **yes** |
| Pendulum-V (12–17 h) | no — 3 chained jobs, needs resume |

Fair-use caveat: GitHub's terms cover Actions for building/testing/deploying the project
and reserve the right to throttle. A few jobs for your own thesis repo is defensible; a
permanent 20-wide training farm is not. Keep it to bursts of ~4–6.

### Kaggle

~30 GPU-h/week (floating quota), **12 h per session**, P100 16 GB or dual T4.
"Save & Run All" commits and runs **headless** — laptop can be closed.

No Docker, and the base image is Python 3.11. Install a 3.8 interpreter alongside rather
than porting the code — **micromamba, not conda** (conda's solver would eat 10 min of a
12 h session):

```bash
!curl -Ls micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
!./bin/micromamba create -y -p /kaggle/working/py38 -c conda-forge python=3.8
!/kaggle/working/py38/bin/pip install torch==1.7.1+cu110 \
    -f https://download.pytorch.org/whl/torch_stable.html
# ...then the same pins as Dockerfile
```

**Unverified risk:** `torch==1.7.1+cu110` on Kaggle's P100/T4. Version support is fine
(sm_60, sm_75) and drivers are backward compatible, but this has not been tested. Fallback
is `torch==1.13.1+cu117` — still Python 3.8, but a stack drift, so Kaggle would then form
its own comparison set.

Kaggle is **not needed** for the sanity envs; it earns its porting cost only at the BLT
configs (~306 h native per seed).

### Capacity, for planning

| source | weekly capacity (native-x86-hours) |
|---|---|
| Kaggle free (30 GPU-h @ 10–20x native CPU) | ~300–600 |
| GitHub Actions (~5 concurrent, fair-use) | ~600–900 |

Counterintuitively the free CPU farm is competitive with the free GPU, because you get
many slots and this workload barely uses a card.

---

## 8. TODO: resume support

**Not implemented.** Required for Pendulum-V and anything larger, because every free tier
caps sessions (6 h Actions, 12 h Kaggle).

Current state: `policies/learner.py:904` `save_model` writes **only**
`agent.state_dict()`, and `learner.py:400` gates it behind
`_n_env_steps_total > 0.75 * n_env_steps_total`. With `save_interval: 100` over 250
iterations that is **one checkpoint near iteration 200**. `load_model` exists at line 910
and **is never called** — there is no `--resume` flag. A killed run is a lost run.

For **bitwise-exact** resume, all of the following must be captured:

| state | source | saved today? |
|---|---|---|
| weights (actor, critic, 2 targets) | `agent.state_dict()` | yes |
| Adam moments x 2 | `actor_optimizer`, `critic_optimizer` `.state_dict()` | no |
| SAC entropy coefficient | `algo.log_alpha_entropy` + `alpha_entropy_optim` | no |
| replay buffer | 6 numpy arrays + `_top`, `_size` (slice to `_size`) | no |
| step counters | `_n_env_steps_total`, `_n_rollouts_total`, `_n_rl_update_steps_total`, `_n_env_steps_total_last`, `_successes_in_buffer` | no |
| RNG: python / numpy / torch | `random.getstate()`, `np.random.get_state()`, `torch.get_rng_state()` | no |
| RNG: env + action_space | `env.np_random`, `env.action_space.np_random` | no |

The last row is what buys *validity* rather than mere continuity. All of it is
serializable, so bitwise resume is achievable for vector-observation envs. Checkpoint size
is small — Pendulum-V is ~1.2 MB (51,000 used rows x 6 fields x 4 bytes).

Implementation notes:

- **Atomic writes.** Write `ckpt.tmp`, then `os.replace()`. A kill mid-write leaves the
  previous checkpoint intact, which is what makes interruption at an arbitrary moment safe.
- **`utils/logger.py:148` opens the CSV with `open(filename, "w+t")` — it truncates.**
  A resume into the same log directory would erase the earlier rows. Needs append-on-resume.
- `policies/main.py` mints a new timestamped directory per launch, so resume needs
  `--resume <log_dir>` to reuse one.
- **Acceptance test:** run N iterations straight through; separately run N/2, interrupt,
  resume, finish. `diff` the two `progress.csv` files. Identical = correct.

Nothing here touches the algorithm — no change to `sampled_seq_len`,
`num_updates_per_iter`, batch size, or any loss — so the comparison against the paper
stays valid.

---

## 9. Provenance of the numbers

| claim | status |
|---|---|
| per-update timings in §6 | **measured** on this laptop (emulated amd64, torch 1.7.1, 1 thread) |
| update counts, episode lengths | **read** from configs and `envs/*/__init__.py` |
| per-run laptop hours | measured x read |
| phase breakdown of one update | **measured** |
| thread scaling, emulation tax (~2x) | **measured** |
| no multi-GPU support; single-device selection | **read from source** |
| 1.3% A100 utilization | arithmetic on a measured FLOP count |
| **20–40x GPU speedup** | **modelled**, anchored on a README prose claim — weakest figure here |
| 8–16 runs packed per card | **modelled**, untested |
| native x86 ~= laptop/2 | **modelled** — extrapolated from a measured arm64-vs-emulated ratio |
| whole-repo 884 PFLOP / 6.1 years | cost model fitted to 5 measurements, **mean error 21%**, worst +36% |

Two known biases: the `credit/` configs have CNN image encoders the FLOP model ignores
(so Catch is an **underestimate**, and it is already the critical path), and the budget
excludes the paper's ablation sweeps (`RL`/`Encoder`/`Len`/`Inputs`/`Arch` in
`scripts/constants.py`) which would multiply the recurrent rows several-fold.

Measurements were also noisy: the same per-update figure came out 1577, 1628 and 2402 ms
across repeats under background load. An early 63 h/seed estimate for Pendulum-V was
**wrong by 2.8x** because the benchmark ran alongside a live training container. Measure
on an idle machine.
