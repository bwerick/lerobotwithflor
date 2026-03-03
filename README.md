# LeRobot with Flor

Extended LeRobot training utilities with [FlorDB](https://github.com/ucbrise/flor) logging support.

This repo wraps HuggingFace LeRobot training, recording, and teleoperation with structured logging via FlorDB.


## Requirements

- Python 3.10+
- NVIDIA GPU (CUDA installed)

## Setup

Create a working directory and clone this repo alongside lerobot:

```bash
git clone https://github.com/bwerick/lerobotwithflor.git
cd lerobotwithflor
make setup
```

`make setup` will:
1. Create a virtual environment at `.venv/`
2. Clone [HuggingFace LeRobot](https://github.com/huggingface/lerobot.git) into `../lerobot` (if not already present)
3. Install lerobot with `feetech` and `smolvla` extras (editable)
4. Install project requirements from `requirements.txt`

Activate the environment with:
```bash
source .venv/bin/activate
```

## Usage

### Train

```bash
make train
```

Runs `automation/train_with_flor.py` with the configured dataset and output directory.

**Key variables (override on CLI):**

| Variable   | Default                          | Description                        |
|------------|----------------------------------|------------------------------------|
| `DATASET`  | `BarbaricErick/so101_3c_merged`  | HuggingFace dataset repo ID        |
| `BASE_OUT` | `outputs/smolvla_make`           | Output directory prefix            |
| `EXTRA`    | `--steps 40000 --batch_size 128 --num_workers 8 ...` | Extra training args |

Example override:
```bash
make train DATASET=myorg/mydataset EXTRA="--steps 10000 --batch_size 64"
```

### Quick Test Train

```bash
make testtrain
```

Runs a fast smoke-test training run (50 steps, batch size 2) to verify the pipeline works end-to-end.

### Record

```bash
make record
```

Runs `automation/record_with_flor.py` to record demonstrations.

## Makefile Targets

| Target        | Description                                                      |
|---------------|------------------------------------------------------------------|
| `make setup`  | Create venv + clone lerobot (if missing) + editable installs     |
| `make train`  | Run training via `automation/train_with_flor.py`                 |
| `make testtrain` | Run a fast test training (50 steps) to verify the pipeline    |
| `make record` | Run recording via `automation/record_with_flor.py`               |
| `make clean`  | Remove `outputs/` directory                                      |
| `make distclean` | Remove `outputs/` and `.venv/`                               |

## Project Structure

```
lerobotwithflor/
├── automation/
│   ├── train_with_flor.py    # Training entrypoint
│   └── record_with_flor.py   # Recording entrypoint
├── outputs/                  # Training run outputs (created by make setup)
├── requirements.txt
└── Makefile
```
