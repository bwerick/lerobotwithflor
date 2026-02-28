SHELL := /bin/bash

PY ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

# LEROBOT repo
LEROBOT_DIR ?= ../lerobot
LEROBOT_REPO ?= https://github.com/huggingface/lerobot.git

# training defaults (override on CLI)
DATASET ?= BarbaricErick/so101_3c_merged
BASE_OUT ?= outputs/smolvla_make
LOG_EVERY ?= 70
SYS_EVERY ?= 70
EXTRA ?= --steps 40000 --batch_size 128 --num_workers 8 --log_freq 10 --save_freq 5000 --policy.push_to_hub false
# generate timestamp at execution time
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
OUT := $(BASE_OUT)_$(TIMESTAMP)

# fast test training defaults (override on CLI)
SMDATASET ?= BarbaricErick/so101_3c_merged
SMBASE_OUT ?= outputs/test_smolvla_make
SMLOG_EVERY ?= 5
SMSYS_EVERY ?= 5
SMEXTRA ?= --steps 50 --batch_size 2 --num_workers 0 --log_freq 1 --save_freq 999999 --policy.push_to_hub false
# generate timestamp at execution time
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
SMOUT := $(SMBASE_OUT)_$(TIMESTAMP)

.PHONY: help venv clone-lerobot install-lerobot install setup testtrain train record clean distclean

help:
	@echo "Targets:"
	@echo "  make setup         # create venv + clone lerobot (if missing) + editable installs"
	@echo "  make train         # run training via automation/train_with_flor.py"
	@echo "  make record        # run recording via automation/record_with_flor.py (if you have it)"
	@echo "Variables:"
	@echo "  LEROBOT_DIR=../lerobot"
	@echo "  DATASET=... OUT=... EXTRA='...'"

venv:
	@test -d $(VENV) || $(PY) -m venv $(VENV)
	@$(PIP) install -U pip setuptools wheel

clone-lerobot:
	@if [ ! -d "$(LEROBOT_DIR)" ]; then \
		echo "Cloning lerobot into $(LEROBOT_DIR)"; \
		git clone $(LEROBOT_REPO) $(LEROBOT_DIR); \
	else \
		echo "Found lerobot at $(LEROBOT_DIR)"; \
	fi

install-lerobot: venv clone-lerobot
	@# base install
	@$(PIP) install -e $(LEROBOT_DIR)
	@# extras (same as your workflow)
	@$(PIP) install -e "$(LEROBOT_DIR)[feetech]"
	@$(PIP) install -e "$(LEROBOT_DIR)[smolvla]"

install: install-lerobot

setup: install
	@mkdir -p outputs
	@echo ""
	@echo "Setup complete."
	@echo "Activate with: source $(VENV)/bin/activate"
	@echo ""

testtrain: venv
	@$(PYBIN) automation/train_with_flor.py \
		--dataset_repo_id $(DATASET) \
		--output_dir $(SMOUT) \
		--log_every $(SMLOG_EVERY) \
		--sys_every $(SMSYS_EVERY) \
		--extra "$(EXTRA)"

train: venv
	@$(PYBIN) automation/train_with_flor.py \
		--dataset_repo_id $(DATASET) \
		--output_dir $(OUT) \
		--log_every $(LOG_EVERY) \
		--sys_every $(SYS_EVERY) \
		--extra "$(EXTRA)"

record: venv
	@$(PYBIN) automation/record_with_flor.py

clean:
	@rm -rf outputs

distclean: clean
	@rm -rf $(VENV)