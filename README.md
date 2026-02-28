# lerobotwithflor

Extended LeRobot training utilities with FlorDB logging support.

This repo wraps HuggingFace LeRobot training, recording, and teleoperation
with structured logging via FlorDB.

---

# Requirements


- Python 3.10+
- NVIDIA GPU (CUDA installed)
- git
- make

---

# Setup

Create a working directory (example: `~/git`):

```bash
mkdir -p ~/git
git clone https://github.com/bwerick/lerobotwithflor.git
cd lerobotwithflor
make setup

make train

