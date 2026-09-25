# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Reference implementation for *Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs* (ICML 2022, Ni/Eysenbach/Salakhutdinov). Recurrent (LSTM/GRU) off-policy actor-critic — TD3, SAC, SAC-discrete — evaluated across POMDP subareas: "standard" POMDPs (occluded PyBullet), meta RL, robust RL, generalization, temporal credit assignment.

This checkout is a replication effort. See `readme.md` for the full per-benchmark command list and `docs/our_details.md` for what each config knob maps to in the paper.

## Environment setup

`environments.yml` is pinned to **linux-64** (`cudatoolkit=11.0`, `gcc_linux-64`, `mujoco-py`) and will not solve on macOS/arm64. On mac, build a minimal env by hand instead — the sanity-check envs (Pendulum-V, CartPole-V) only need `python=3.7–3.9`, `torch`, `gym~=0.21`, `pybullet`/`pybullet_envs`, `ruamel.yaml`, `absl-py`, `psutil`, `tensorboard`, `matplotlib`, `future_fstrings`. MuJoCo is only needed for `configs/meta/*mujoco*`; Roboschool (docker/singularity `.sif`) only for `configs/rmdp` and `configs/generalize`.

The code targets the old `gym` API (`env.seed(...)`, 4-tuple `step`, `Pendulum-v1`, `CartPole-v0`, `LunarLander-v2`). Do not "upgrade" to gymnasium piecemeal — `envs/pomdp/wrappers.py`, `envs/meta/wrappers.py`, and `policies/learner.py` all assume the legacy API.

Run Jupyter from inside the project env, not the Homebrew `jupyterlab` (its tornado 6.5.9 + jupyter_server 2.21 combination raises `AttributeError: 'FileFindHandler' object has no attribute 'allowed_symlink_directory'` on every static asset).

## Commands

Every entry point assumes the repo root is on `PYTHONPATH`:

```bash
export PYTHONPATH=${PWD}:$PYTHONPATH
```

Training (single entry point for all subareas — there is no test suite, no linter config):

```bash
# smallest sanity check: Pendulum-V, ~50k env steps
python policies/main.py --cfg configs/pomdp/pendulum/v/rnn.yml --algo td3

# --debug writes to debug/ instead of logs/; shorten runs by editing train.num_iters
python policies/main.py --cfg configs/pomdp/pendulum/v/rnn.yml --algo sac --debug --seed 0 --cuda -1
```

CLI flags override the yaml: `--env --algo {td3,sac,sacd} --seed --cuda --oracle --(no)automatic_entropy_tuning --target_entropy --entropy_alpha --debug`. `--cuda -1` forces CPU. `--oracle` feeds privileged state (reduces the POMDP to its MDP) and is the paper's upper baseline; `mlp.yml` configs without `--oracle` are the Markovian lower baseline.

Formatting (the only enforced convention): `black . -t py35`.

Plotting a run — see `docs/plot_curves.md`; `scripts/eval.sh` reproduces every figure in the paper:

```bash
mkdir -p results && cp -r logs/ results/
python scripts/merge_csv.py --base_path results/logs/<subarea>/<env_name>   # -> results/data/.../final.csv
python scripts/plot_csv.py --csv_path results/data/<subarea>/<env_name>/final.csv --best_variant <name>
```

New methods/environments must be registered in `scripts/constants.py` for the plotting scripts to see them.

## Architecture

**`policies/main.py`** — argparse/yaml merge, seeding, then builds a nested log directory *from the hyperparameters themselves*: `logs/<env_type>/<env>/<pomdp_type>/<algo>_<seq_model>/gamma-*/len-*/bs-*/freq-*/<oar>/<timestamp>/`. That path encoding is what `scripts/merge_csv.py` parses back out, so changing the naming breaks plotting. It also snapshots `policies/` and the resolved config into the log dir.

**`policies/learner.py`** — the whole training loop and all env-family branching. `init_env` dispatches on `env.env_type` ∈ `{pomdp, credit, meta, rmdp, generalize, atari}`, each with a different wrapper stack and a different notion of "task"; `init_agent` picks the policy class; `init_train` picks the buffer. Loop is: `num_init_rollouts_pool` random rollouts → per iteration, collect `num_rollouts_per_iter` episodes, then `num_updates_per_iter` gradient steps (an `int` is absolute; a `float` is a multiplier on env steps collected). Adding an environment family means adding a branch here, not just a config.

**Three orthogonal axes** — do not conflate them:
- *Architecture* (`policies/models/`, selected by `policy.seq_model` + `policy.separate`): `Policy_MLP` (Markovian), `Policy_Separate_RNN` (paper's best: separate actor and critic RNNs), `Policy_Shared_RNN`, `Policy_RNN_MLP` (`lstm-mlp`/`gru-mlp`). Exposed as `AGENT_ARCHS` ∈ `{Markov, Memory, Memory_Markov}`, which is what the learner branches on for rollout/buffer handling.
- *RL algorithm* (`policies/rl/`, `RL_ALGORITHMS` keyed by `policy.algo_name`): `td3`, `sac`, `sacd`. Each supplies `build_actor` / `build_critic` / `select_action` / `critic_loss` / `actor_loss`, so the recurrent architectures are algorithm-agnostic. Per-algo hyperparameters live in a nested `policy.<algo_name>` block in the yaml and are forwarded as `**kwargs[algo_name]`.
- *Buffer* (`buffers/`): `SimpleReplayBuffer` for Markovian; `SeqReplayBuffer` (vanilla, stores obs twice — fine for vector obs) or `RAMEfficient_SeqReplayBuffer` (for pixel obs, chosen via `train.buffer_type`) for recurrent. Sequence buffers are flat 2-D arrays with episode bookkeeping and sample `(sampled_seq_len, batch_size, dim)` subsequences; `sampled_seq_len: -1` means the full episode.

**Config = paper's decision factors.** `policy.separate` (Arch), `policy.seq_model` (Encoder), `policy.algo_name` (RL), `train.sampled_seq_len` (Len), and the `*_embedding_size` fields (Inputs — setting `action_embedding_size` or `reward_embedding_size` to `0` *removes* past actions/rewards from the policy input; the `o`/`oa`/`oar` suffix in the log path records this). Reproducing a paper number means matching these five plus `gamma`, `batch_size`, and `num_updates_per_iter`.

**Environment registration.** `envs/pomdp/__init__.py` registers `<Name>-{F,P,V}-v0` via `gym.register` (F = fully observed, P = position/angle only, V = velocity only), wrapping a base gym/PyBullet env with `POMDPWrapper(partially_obs_dims=[...])`. Importing `envs.pomdp` imports `pybullet_envs`, so PyBullet is a hard dependency even for Pendulum. Meta-RL envs go through `envs/meta/make_env.py` (VariBAD-style wrapper, `horizon_bamdp = k * H`).

**Metrics.** `utils/logger.py` writes stdout + `progress.csv` + tensorboard keyed on total env steps. `metrics/return_eval_total` is the headline learning-curve number; `rl_loss/*` are the training diagnostics; `z/*` are step counters.

## Known staleness

`example.ipynb` (the README's "minimal example") predates the camera-ready refactor and **fails as written**: cell 4 passes `state_embedding_size=` and `algo=` to `Policy_RNN`, which now expects `observ_embedding_size=` and `algo_name=`, and `algo_name` requires a nested per-algorithm kwarg dict (e.g. `td3=dict(exploration_noise=0.1, target_noise=0.2, target_noise_clip=0.5)`). `docs/our_details.md` also links `buffers/seq_replay_buffer.py`, which was split into the `_vanilla` / `_efficient` files. Prefer `policies/main.py --cfg configs/pomdp/pendulum/v/rnn.yml` over the notebook for a first cross-validation run; fix the notebook against the current `policies/models/policy_rnn.py` signature if it is needed.
