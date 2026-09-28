# -*- coding: future_fstrings -*-
"""Bitwise-exact checkpoint/resume for Learner.

Every free compute tier caps session length (6 h on GitHub Actions, 12 h on
Kaggle), and a killed run is otherwise a lost run: learner.save_model() stores
only agent.state_dict(), gated behind 75% of training, and load_model is never
called.

A resume is only *valid* if it reproduces the numbers an uninterrupted run would
have produced. That needs more than weights -- it needs the optimizer moments,
the entropy coefficient, the replay buffer, the step counters, and every random
number stream. All five are captured here, so
    run N iterations
and
    run N/2, kill, resume, run N/2
produce identical progress.csv files.

Writes are atomic (tmp file + os.replace), so an interruption during a write
cannot corrupt the previous checkpoint.
"""
import os
import random

import numpy as np
import torch

from torchkit import pytorch_utils as ptu
from utils import logger

# bumped if the payload layout changes in a way old files cannot satisfy
FORMAT_VERSION = 1

_COUNTERS = (
    "_n_env_steps_total",
    "_n_env_steps_total_last",
    "_n_rl_update_steps_total",
    "_n_rollouts_total",
    "_successes_in_buffer",
)

# buffer arrays are sliced to _size: when the buffer has wrapped, _size ==
# max_replay_buffer_size and the slice is the whole array, so this is always correct
_BUFFER_ARRAYS = (
    "_observations",
    "_next_observations",
    "_actions",
    "_rewards",
    "_terminals",
    "_valid_starts",
)


def _np_random_state(obj):
    """gym 0.21 exposes np.random.RandomState as .np_random on envs and spaces."""
    rng = getattr(obj, "np_random", None)
    if rng is None:
        return None
    if hasattr(rng, "get_state"):
        return rng.get_state()
    return None


def _set_np_random_state(obj, state):
    if state is None:
        return
    rng = getattr(obj, "np_random", None)
    if rng is not None and hasattr(rng, "set_state"):
        rng.set_state(state)


def _csv_path(ckpt_path):
    return os.path.join(os.path.dirname(ckpt_path), "progress.csv")


def _count_csv_rows(ckpt_path):
    """Data rows (excluding header) currently in progress.csv."""
    csv = _csv_path(ckpt_path)
    if not os.path.exists(csv):
        return 0
    with open(csv) as f:
        return max(0, sum(1 for _ in f) - 1)


def truncate_csv(ckpt_path):
    """Drop progress.csv rows written AFTER the checkpoint was taken.

    A run killed between two checkpoints has already logged rows for iterations
    the resumed run will redo. Without this the appended curve contains duplicate
    steps, which merge_csv would read as real data. Called from main.py before the
    logger opens the file.
    """
    marker = ckpt_path + ".rows"
    if not os.path.exists(marker):
        return None
    with open(marker) as f:
        keep = int(f.read().strip())
    csv = _csv_path(ckpt_path)
    if not os.path.exists(csv):
        return None
    with open(csv) as f:
        lines = f.readlines()
    if len(lines) <= keep + 1:
        return 0
    dropped = len(lines) - (keep + 1)
    with open(csv, "w") as f:
        f.writelines(lines[: keep + 1])
    return dropped


def save(learner, path, last_eval_num_iters=0):
    """Atomically write the full resumable state of `learner` to `path`."""
    agent = learner.agent
    algo = agent.algo
    buf = learner.policy_storage
    size = buf._size

    payload = {
        "format_version": FORMAT_VERSION,
        "seed": learner.seed,
        # --- networks and optimizers -------------------------------------
        "agent": agent.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        # --- SAC/SAC-discrete learned entropy coefficient ----------------
        "automatic_entropy_tuning": getattr(algo, "automatic_entropy_tuning", False),
        # --- replay buffer ------------------------------------------------
        "buffer": {k: getattr(buf, k)[:size].copy() for k in _BUFFER_ARRAYS},
        "buffer_top": buf._top,
        "buffer_size": size,
        # --- progress counters -------------------------------------------
        "counters": {k: getattr(learner, k) for k in _COUNTERS},
        "last_eval_num_iters": last_eval_num_iters,
        # keep z/time_cost cumulative across segments rather than per-segment
        "elapsed_seconds": learner._elapsed_seconds(),
        # --- every random stream -----------------------------------------
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "rng_train_env": _np_random_state(learner.train_env),
        "rng_train_action_space": _np_random_state(learner.train_env.action_space),
        "rng_eval_env": (
            None
            if learner.eval_env is learner.train_env
            else _np_random_state(learner.eval_env)
        ),
    }
    if payload["automatic_entropy_tuning"]:
        payload["log_alpha_entropy"] = algo.log_alpha_entropy.detach().cpu().clone()
        payload["alpha_entropy_optim"] = algo.alpha_entropy_optim.state_dict()

    rows = _count_csv_rows(path)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on POSIX: a kill mid-write keeps the old file
    # Written after the .pt so the row count can never claim more than the
    # checkpoint covers; a stale-low value only costs a few redone rows.
    tmp_rows = path + ".rows.tmp"
    with open(tmp_rows, "w") as f:
        f.write(str(rows))
    os.replace(tmp_rows, path + ".rows")
    return path


def load(learner, path):
    """Restore `learner` in place. Returns last_eval_num_iters."""
    payload = torch.load(path, map_location=ptu.device)

    got = payload.get("format_version")
    if got != FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format {got} != expected {FORMAT_VERSION}: {path}"
        )
    if payload["seed"] != learner.seed:
        raise ValueError(
            f"checkpoint seed {payload['seed']} != --seed {learner.seed}. Resuming "
            f"with a different seed would silently invalidate the run."
        )

    agent = learner.agent
    algo = agent.algo
    agent.load_state_dict(payload["agent"])
    agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])

    if payload["automatic_entropy_tuning"]:
        with torch.no_grad():
            algo.log_alpha_entropy.copy_(payload["log_alpha_entropy"].to(ptu.device))
        algo.alpha_entropy_optim.load_state_dict(payload["alpha_entropy_optim"])
        algo.alpha_entropy = algo.log_alpha_entropy.exp().detach().item()

    buf = learner.policy_storage
    size = payload["buffer_size"]
    for k, arr in payload["buffer"].items():
        getattr(buf, k)[:size] = arr
    buf._top = payload["buffer_top"]
    buf._size = size

    for k, v in payload["counters"].items():
        setattr(learner, k, v)

    random.setstate(payload["rng_python"])
    np.random.set_state(payload["rng_numpy"])
    torch.set_rng_state(payload["rng_torch"])
    if payload["rng_torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["rng_torch_cuda"])
    _set_np_random_state(learner.train_env, payload["rng_train_env"])
    _set_np_random_state(
        learner.train_env.action_space, payload["rng_train_action_space"]
    )
    if payload["rng_eval_env"] is not None:
        _set_np_random_state(learner.eval_env, payload["rng_eval_env"])

    learner._restore_elapsed(payload["elapsed_seconds"])
    logger.log(
        f"resumed from {path}: env_steps {learner._n_env_steps_total}, "
        f"rl_steps {learner._n_rl_update_steps_total}, buffer {size}"
    )
    return payload["last_eval_num_iters"]
