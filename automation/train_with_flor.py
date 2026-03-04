#!/usr/bin/env python3
"""
automation/train_with_flor.py

Fine-tune (or train) a LeRobot policy (default: SmolVLA) while logging:
- run config / hyperparams (once)
- training and system metrics

Cross-platform:
- psutil is optional but recommended for CPU/RAM metrics
- NVIDIA GPU util% is optional (pynvml). CUDA mem stats via torch if available.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional, Tuple
import threading
import flordb as flor

# Optional deps
try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import torch  # type: ignore
except Exception:
    torch = None


# ----------------------------
# Helpers: git / system info
# TODO: Makes sense to put these in a shared utils module if we have multiple scripts.
# ----------------------------


_NVML_READY = False
_NVML_HANDLE = None


def init_nvml(device_index: int = 0) -> None:
    global _NVML_READY, _NVML_HANDLE
    if _NVML_READY:
        return
    try:
        import pynvml

        pynvml.nvmlInit()
        _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        _NVML_READY = True
    except Exception:
        _NVML_READY = False
        _NVML_HANDLE = None


def shutdown_nvml() -> None:
    global _NVML_READY, _NVML_HANDLE
    if not _NVML_READY:
        return
    try:
        import pynvml

        pynvml.nvmlShutdown()
    except Exception:
        pass
    _NVML_READY = False
    _NVML_HANDLE = None


def try_get_gpu_metrics() -> None:
    """
    Requires `nvidia-ml-py` (imports as `pynvml`).
    """
    if not _NVML_READY or _NVML_HANDLE is None:
        return

    try:
        h = _NVML_HANDLE

        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")

        mem = pynvml.nvmlDeviceGetMemoryInfo(h)  # bytes
        util = pynvml.nvmlDeviceGetUtilizationRates(h)  # %
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)

        out = {
            "gpu/name": name,
            "gpu/util_pct": int(util.gpu),
            "gpu/mem_util_pct": int(util.memory),
            "gpu/mem_used_mb": round(mem.used / (1024**2), 1),
            "gpu/mem_total_mb": round(mem.total / (1024**2), 1),
            "gpu/mem_free_mb": round(mem.free / (1024**2), 1),
            "gpu/temp_c": int(temp),
        }

        # optional: power (not always supported)
        try:
            mw = pynvml.nvmlDeviceGetPowerUsage(h)  # milliwatts
            out["gpu/power_w"] = round(mw / 1000.0, 1)
        except Exception:
            pass

        # optional: clocks
        try:
            out["gpu/sm_clock_mhz"] = int(
                pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            )
            out["gpu/mem_clock_mhz"] = int(
                pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
            )
        except Exception:
            pass

        for k, v in out.items():
            flor.log(k, v)

    except Exception:
        pass


def log_static_sys_info() -> None:
    # flor.dataframe("sys/%")
    info: Dict[str, Any] = {
        "sys/os": platform.platform(),
        "sys/machine": platform.machine(),
        "sys/python": platform.python_version(),
        "sys/executable": sys.executable,
    }

    # if torch is not defined or

    if torch is not None:
        info["sys/torch_version"] = getattr(torch, "__version__", None)
        try:
            info["sys/cuda_available"] = bool(torch.cuda.is_available())
        except Exception:
            info["sys/cuda_available"] = None

        if info.get("sys/cuda_available"):
            try:
                info["sys/cuda_version"] = getattr(torch.version, "cuda", None)
            except Exception:
                info["sys/cuda_version"] = None
            try:
                info["sys/gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                info["sys/gpu_name"] = None
        else:
            info["sys/cuda_version"] = None
            info["sys/gpu_name"] = None
    else:
        info["sys/torch_version"] = None
        info["sys/cuda_available"] = None
        info["sys/cuda_version"] = None
        info["sys/gpu_name"] = None

    info["sys/psutil_available"] = psutil is not None
    info["sys/pynvml_available"] = pynvml is not None

    for k, v in info.items():
        flor.log(k, v)


def get_dynamic_sys_metrics() -> Dict[str, Any]:
    """
    Dynamic metrics (sampled repeatedly).
    Keep column names stable.
    """
    m: Dict[str, Any] = {}

    # CPU/RAM (psutil)
    if psutil is not None:
        try:
            m["sys/cpu_pct"] = float(psutil.cpu_percent(interval=None))
        except Exception:
            m["sys/cpu_pct"] = None
        try:
            vm = psutil.virtual_memory()
            m["sys/ram_used_gb"] = float((vm.total - vm.available) / (1024**3))
            m["sys/ram_pct"] = float(vm.percent)
        except Exception:
            m["sys/ram_used_gb"] = None
            m["sys/ram_pct"] = None
    else:
        m["sys/cpu_pct"] = None
        m["sys/ram_used_gb"] = None
        m["sys/ram_pct"] = None

    # GPU memory (torch) + GPU util (pynvml, optional)
    if torch is not None:
        try:
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
    else:
        cuda_ok = False

    if cuda_ok and torch is not None:
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            used_b = total_b - free_b
            m["sys/gpu_mem_used_gb"] = float(used_b / (1024**3))
            m["sys/gpu_mem_total_gb"] = float(total_b / (1024**3))
        except Exception:
            m["sys/gpu_mem_used_gb"] = None
            m["sys/gpu_mem_total_gb"] = None
    else:
        m["sys/gpu_mem_used_gb"] = None
        m["sys/gpu_mem_total_gb"] = None

    # NVIDIA util% (optional; only meaningful on NVIDIA + pynvml)
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            m["sys/gpu_util_pct"] = float(util.gpu)
        except Exception:
            m["sys/gpu_util_pct"] = None
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    else:
        m["sys/gpu_util_pct"] = None

    return m


# ---------------------------------------
# LeRobot log parsing: extract key metrics
# ---------------------------------------

_METRIC_PATTERNS = [
    # Common "key=value" patterns
    re.compile(r"\b(step|global_step)\s*=\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(epoch)\s*=\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(loss)\s*=\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\b"),
    re.compile(
        r"\b(lr|learning_rate)\s*=\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\b",
        re.IGNORECASE,
    ),
    # JSON-ish metrics: "metric/loss": 0.123, etc.
    re.compile(
        r'"metric/(step|epoch|loss|lr)"\s*:\s*("?)([0-9]*\.?[0-9]+)\2', re.IGNORECASE
    ),
]


def parse_metrics_from_line(line: str) -> Dict[str, Any]:
    """
    Extract metrics from a single stdout line.
    Returns stable keys: train/step, train/epoch, train/loss, train/lr (when found).
    """
    out: Dict[str, Any] = {}

    for pat in _METRIC_PATTERNS:
        for m in pat.finditer(line):
            if pat.pattern.startswith('"metric/'):
                key = m.group(1).lower()
                val = m.group(3)
            else:
                key = m.group(1).lower()
                val = m.group(2)

            if key in ("global_step", "step"):
                out["train/step"] = int(float(val))
            elif key == "epoch":
                out["train/epoch"] = int(float(val))
            elif key == "loss":
                out["train/loss"] = float(val)
            elif key in ("lr", "learning_rate"):
                out["train/lr"] = float(val)

    return out


# ----------------------------
# Config model
# ----------------------------


@dataclass
class TrainRunConfig:
    # What to train
    policy_type: str = "smolvla"
    dataset_repo_id: str = ""  # e.g. "BarbaricErick/my_merged_dataset"
    output_dir: str = "outputs/train"
    # Logging cadence
    log_every: int = 50  # log train metrics every N steps
    sys_every: int = 50  # log sys metrics every M steps (often same as log_every)
    stdout_every: int = 0  # 0 disables; otherwise log every K lines
    # Optional tags
    run_tag: str = ""
    seed: int = 0
    # Pass-through extra args for lerobot-train
    extra: Optional[str] = None  # extra CLI args string


# ----------------------------
# Main training runner
# ----------------------------


def build_lerobot_train_cmd(cfg: TrainRunConfig) -> list[str]:
    """
    Builds a lerobot training command. Adjust flags to match your actual CLI usage.
    """
    if not cfg.dataset_repo_id:
        raise ValueError(
            "dataset_repo_id is required (HF repo id for LeRobot dataset)."
        )

    cmd = [
        "lerobot-train",
        "--policy.type",
        cfg.policy_type,
        "--dataset.repo_id",
        cfg.dataset_repo_id,
        "--output_dir",
        cfg.output_dir,
    ]

    # Optional: seed
    if cfg.seed:
        cmd += ["--seed", str(cfg.seed)]

    # Optional: extra CLI args string
    if cfg.extra:
        # split respecting quotes
        cmd += shlex.split(cfg.extra)

    return cmd


def run_train(cfg: TrainRunConfig) -> int:
    # -------------------------
    # Run-level Flor logging
    # -------------------------

    # system metadata as logs
    log_static_sys_info()

    cmd = build_lerobot_train_cmd(cfg)
    flor.log("train/cmd", " ".join(shlex.quote(x) for x in cmd))

    # try to log gpu metrics
    init_nvml(device_index=0)
    try_get_gpu_metrics()

    # -------------------------
    # Start process
    # -------------------------

    p = subprocess.Popen(cmd)
    # run the process
    p.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--policy_type",
        default="smolvla",
        help="LeRobot policy type (default: smolvla)",
    )
    ap.add_argument(
        "--dataset_repo_id",
        required=True,
        help="HF dataset repo id, e.g. BarbaricErick/my_dataset",
    )
    ap.add_argument("--output_dir", default="outputs/train", help="Training output dir")
    ap.add_argument(
        "--log_every", type=int, default=50, help="Log training metrics every N steps"
    )
    ap.add_argument(
        "--sys_every", type=int, default=50, help="Log system metrics every M steps"
    )
    ap.add_argument(
        "--stdout_every",
        type=int,
        default=0,
        help="Log stdout every K lines (0 disables)",
    )
    ap.add_argument("--run_tag", default="", help="Optional run tag")
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (if supported by your training)",
    )
    ap.add_argument(
        "--extra",
        default=None,
        help="Extra args passed through to lerobot-train (quoted string)",
    )

    args = ap.parse_args()

    cfg = TrainRunConfig(
        policy_type=flor.arg("policy_type", args.policy_type),
        dataset_repo_id=flor.arg("dataset_repo_id", args.dataset_repo_id),
        output_dir=args.output_dir,
        log_every=args.log_every,
        sys_every=args.sys_every,
        stdout_every=args.stdout_every,
        run_tag=flor.arg("run_tag", args.run_tag),
        seed=flor.arg("seed", args.seed),
        extra=flor.arg("extra", args.extra),
    )

    return run_train(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
