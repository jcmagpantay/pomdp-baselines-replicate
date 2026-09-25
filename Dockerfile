# Replication environment for "Recurrent Model-Free RL Can Be a Strong Baseline
# for Many POMDPs" (ICML 2022).
#
# environments.yml is a linux-64 conda export and cannot solve on arm64. This
# image reproduces every pinned version that touches the computation, so the repo
# runs with ZERO source edits. Each pin is load-bearing:
#   python 3.8      utils/logger.py:9 `from collections import Set` (gone in 3.10)
#   torch 1.7.1     no arm64 build of 1.7.1 exists at all
#   gym 0.21.0      legacy 4-tuple step() + env.seed(); see policies/learner.py
#   numpy 1.20.1    torchkit/pytorch_utils.py:73 uses np.bool (gone in 1.24)
#   pillow 7.2.0    newer Pillow needs numpy.typing.NDArray, absent in 1.20
#   box2d 2.3.10    envs/pomdp/__init__.py builds LunarLander at import time,
#                   so Box2D is needed even for a Pendulum-only run
#   protobuf 3.19.4 tensorboard 2.8's generated _pb2 files reject protobuf>=4
#
# Omitted deliberately: cudatoolkit, mujoco-py, Roboschool, dm-control, Atari
# ROMs. policies/learner.py:init_env imports none of them for env_type=pomdp.
#
# Build:  docker build --platform linux/amd64 -t pomdp-repl .
# Run:    docker run --rm -it -v "$PWD":/workspace pomdp-repl \
#           python policies/main.py --cfg configs/pomdp/pendulum/v/rnn.yml \
#           --algo sac --cuda -1
FROM --platform=linux/amd64 python:3.8-slim-bullseye

# gym 0.21's setup.py fails under modern setuptools ("'extras_require' must be a
# dictionary"), so the build toolchain is pinned back first.
RUN pip install --no-cache-dir pip==22.0.4 setuptools==65.5.0 wheel==0.38.4

# Everything below resolves to a prebuilt wheel: no compiler, no apt layer.
# Debian bullseye/buster archives now 404, so avoiding apt entirely is also what
# keeps this image buildable.
RUN pip install --no-cache-dir --only-binary=:all: \
      torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir --only-binary=:all: \
      numpy==1.20.1 \
      pillow==7.2.0 \
      pybullet==3.1.8 \
      box2d==2.3.10 \
      ruamel.yaml==0.16.12 \
      absl-py==0.11.0 \
      future-fstrings==1.2.0 \
      tensorboard==2.8.0 \
      tensorboardx==1.8 \
      protobuf==3.19.4 \
      psutil==5.8.0 \
      matplotlib==3.3.4 \
      pandas==1.2.2 \
      seaborn==0.11.1 \
      scikit-learn==0.23.2 \
      scipy==1.6.0 \
      joblib==1.0.1 \
      opencv-python==4.5.1.48 \
      dm-env==1.5

# policies/learner.py:105 imports envs.credit_assign unconditionally on the
# pomdp branch, so the Key-to-Door deps are required even for Pendulum.
RUN pip install --no-cache-dir pycolab==1.2

# gym 0.21.0 is sdist-only and must build under the pinned setuptools above.
RUN pip install --no-cache-dir gym==0.21.0

WORKDIR /workspace
ENV PYTHONPATH=/workspace
