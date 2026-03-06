SHELL := /bin/bash

PY ?= python3
PY_GROOT ?= python3.10

VENV ?= .venv
VENV_GROOT ?= .venv-groot

PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

PIP_GROOT := $(VENV_GROOT)/bin/pip
PYBIN_GROOT := $(VENV_GROOT)/bin/python

# LEROBOT repo
LEROBOT_DIR ?= ../lerobot
LEROBOT_REPO ?= https://github.com/huggingface/lerobot.git

# ---------- SmolVLA training defaults ----------
DATASET ?= BarbaricErick/so101_3c_merged
BASE_OUT ?= outputs/smolvla_make
LOG_EVERY ?= 70
SYS_EVERY ?= 70
EXTRA ?= --steps 40000 --batch_size 32 --num_workers 8 --log_freq 10 --save_freq 5000 --policy.push_to_hub false

# ---------- SmolVLA fast test defaults ----------
SMDATASET ?= BarbaricErick/so101_3c_merged
SMBASE_OUT ?= outputs/test_smolvla_make
SMLOG_EVERY ?= 5
SMSYS_EVERY ?= 5
SMEXTRA ?= --steps 128 --batch_size 16 --num_workers 1 --log_freq 1 --save_freq 999999 --policy.push_to_hub false

# ---------- GR00T defaults ----------
GROOT_DATASET ?= BarbaricErick/so101_3c_merged
GROOT_BASE_OUT ?= outputs/groot_make
GROOT_LOG_EVERY ?= 20
GROOT_SYS_EVERY ?= 20
GROOT_EXTRA ?= --steps 10000 --batch_size 1 --num_workers 2 --log_freq 10 --save_freq 1000 --policy.push_to_hub false

TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
OUT := $(BASE_OUT)_$(TIMESTAMP)
SMOUT := $(SMBASE_OUT)_$(TIMESTAMP)
GROOT_OUT := $(GROOT_BASE_OUT)_$(TIMESTAMP)

.PHONY: help venv venv-groot clone-lerobot install-reqs install-lerobot install \
        install-lerobot-groot install-groot setup setup-groot \
        testtrain train train-groot record clean distclean distclean-groot

help:
	@echo "Targets:"
	@echo "  make setup              # setup base .venv for SmolVLA"
	@echo "  make setup-groot        # setup .venv-groot for GR00T"
	@echo "  make train              # run SmolVLA training via FlorCL wrapper"
	@echo "  make testtrain          # run short SmolVLA test via FlorCL wrapper"
	@echo "  make train-groot        # run GR00T training via FlorCL wrapper"
	@echo "  make record             # run recording via Flor wrapper"

venv:
	@test -d $(VENV) || $(PY) -m venv $(VENV)
	@$(PIP) install -U pip wheel
	@$(PIP) install "setuptools>=71,<81"

venv-groot:
	@test -d $(VENV_GROOT) || $(PY_GROOT) -m venv $(VENV_GROOT)
	@$(PIP_GROOT) install -U pip wheel
	@$(PIP_GROOT) install "setuptools>=71,<81"

clone-lerobot:
	@if [ ! -d "$(LEROBOT_DIR)" ]; then \
		echo "Cloning lerobot into $(LEROBOT_DIR)"; \
		git clone $(LEROBOT_REPO) $(LEROBOT_DIR); \
	else \
		echo "Found lerobot at $(LEROBOT_DIR)"; \
	fi

install-reqs: venv
	@echo "Installing project requirements..."
	@$(PIP) install -r requirements.txt

install-reqs-groot: venv-groot
	@echo "Installing project requirements into GR00T env..."
	@$(PIP_GROOT) install -r requirements.txt

install-lerobot: venv clone-lerobot
	@$(PIP) install -e $(LEROBOT_DIR)
	@$(PIP) install -e "$(LEROBOT_DIR)[feetech]"
	@$(PIP) install -e "$(LEROBOT_DIR)[smolvla]"

install-lerobot-groot: venv-groot clone-lerobot
	@$(PIP_GROOT) install "torch>=2.2.1,<2.8.0" "torchvision>=0.21.0,<0.23.0"
	@$(PIP_GROOT) install ninja "packaging>=24.2,<26.0" psutil einops
	@MAX_JOBS=4 $(PIP_GROOT) install "flash-attn==2.7.1.post4" --no-build-isolation
	@$(PIP_GROOT) install -e $(LEROBOT_DIR)
	@$(PIP_GROOT) install -e "$(LEROBOT_DIR)[feetech]"
	@$(PIP_GROOT) install -e "$(LEROBOT_DIR)[groot]"

install: install-lerobot install-reqs
install-groot: install-lerobot-groot install-reqs-groot

setup: install
	@mkdir -p outputs
	@echo ""
	@echo "Base setup complete."
	@echo "Activate with: source $(VENV)/bin/activate"
	@echo ""

setup-groot: install-groot
	@mkdir -p outputs
	@echo ""
	@echo "GR00T setup complete."
	@echo "Activate with: source $(VENV_GROOT)/bin/activate"
	@echo ""

testtrain: venv
	@$(PYBIN) automation/train_with_florcl.py \
		--policy_type smolvla \
		--dataset_repo_id $(SMDATASET) \
		--output_dir $(SMOUT) \
		--log_every $(SMLOG_EVERY) \
		--sys_every $(SMSYS_EVERY) \
		--extra "$(SMEXTRA)"

train: venv
	@$(PYBIN) automation/train_with_florcl.py \
		--policy_type smolvla \
		--dataset_repo_id $(DATASET) \
		--output_dir $(OUT) \
		--log_every $(LOG_EVERY) \
		--sys_every $(SYS_EVERY) \
		--extra "$(EXTRA)"

train-groot: venv-groot
	@$(PYBIN_GROOT) automation/train_with_florcl.py \
		--policy_type groot \
		--dataset_repo_id $(GROOT_DATASET) \
		--output_dir $(GROOT_OUT) \
		--log_every $(GROOT_LOG_EVERY) \
		--sys_every $(GROOT_SYS_EVERY) \
		--extra "$(GROOT_EXTRA)"

record: venv
	@$(PYBIN) automation/record_with_flor.py

clean:
	@rm -rf outputs

distclean: clean
	@rm -rf $(VENV)

distclean-groot:
	@rm -rf $(VENV_GROOT)