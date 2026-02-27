#!/usr/bin/env python3
"""
automation/train_with_flor.py

Fine-tune (or train) a LeRobot policy (default: SmolVLA) while logging:
- run config / hyperparams (once) via flor.arg(...)
- training metrics (every N steps) via flor.loop("step", step) + flor.log(...)
- system metrics (every M steps) via flor.loop("step", step) + flor.log(...)
- optional stdout lines (stable column names; no per-line columns)

Key FlorDB rule:
  flor.log(<NAME>, <VALUE>)  -> <NAME> becomes a COLUMN in the pivoted view.
So <NAME> must be STABLE (e.g., "train/loss"), NOT "train/stdout_line/000123".
Repetition is represented using flor.loop(...).

Cross-platform:
- psutil is optional but recommended for CPU/RAM metrics
- NVIDIA GPU util% is optional (pynvml). CUDA mem stats via torch if available.

Run inside a Git repo (Flor expects that).
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

from typing import Optional, Dict, Any

# Optional deps
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None


# ----------------------------
# Helpers: git / system info
# ----------------------------

# ---- gpu_metrics.py-ish (module-level) ----


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

def try_get_gpu_metrics() -> Dict[str, Any]:
    """
    Returns dict with GPU metrics or {} if unavailable.
    Requires `nvidia-ml-py` (imports as `pynvml`).
    """
    if not _NVML_READY or _NVML_HANDLE is None:
        return {}

    try:
        import pynvml
        h = _NVML_HANDLE

        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")

        mem = pynvml.nvmlDeviceGetMemoryInfo(h)          # bytes
        util = pynvml.nvmlDeviceGetUtilizationRates(h)   # %
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
            out["gpu/sm_clock_mhz"] = int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
            out["gpu/mem_clock_mhz"] = int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
        except Exception:
            pass

        return out
    except Exception:
        return {}

def _run_cmd(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def get_git_info() -> Dict[str, Any]:
    commit = _run_cmd(["git", "rev-parse", "HEAD"])
    dirty = _run_cmd(["git", "status", "--porcelain"])
    return {
        "git/commit": commit,
        "git/dirty": None if dirty is None else (len(dirty) > 0),
    }


def get_static_sys_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "sys/os": platform.platform(),
        "sys/machine": platform.machine(),
        "sys/python": platform.python_version(),
        "sys/executable": sys.executable,
    }

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

    info.update(get_git_info())
    info["sys/psutil_available"] = psutil is not None
    info["sys/pynvml_available"] = pynvml is not None
    return info


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
            m["sys/ram_used_gb"] = float((vm.total - vm.available) / (1024 ** 3))
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
            m["sys/gpu_mem_used_gb"] = float(used_b / (1024 ** 3))
            m["sys/gpu_mem_total_gb"] = float(total_b / (1024 ** 3))
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
    re.compile(r"\b(lr|learning_rate)\s*=\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\b", re.IGNORECASE),

    # JSON-ish metrics: "metric/loss": 0.123, etc.
    re.compile(r'"metric/(step|epoch|loss|lr)"\s*:\s*("?)([0-9]*\.?[0-9]+)\2', re.IGNORECASE),
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
    dataset_repo_id: str = ""          # e.g. "BarbaricErick/my_merged_dataset"
    output_dir: str = "outputs/train"  # where checkpoints/logs go
    # Logging cadence
    log_every: int = 50                # log train metrics every N steps
    sys_every: int = 50                # log sys metrics every M steps (often same as log_every)
    stdout_every: int = 0              # 0 disables; otherwise log every K lines
    # Optional tags
    run_tag: str = ""
    seed: int = 0
    # Pass-through extra args for lerobot-train
    extra: Optional[str] = None        # extra CLI args string




_SAFE_ARG_RE = re.compile(r"[^A-Za-z0-9_]+")

def _safe_arg_name(name: str) -> str:
    # turn "cfg/batch_size" into "cfg_batch_size"
    name = name.strip()
    name = _SAFE_ARG_RE.sub("_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

def flor_log_config_as_args(cfg_dict: dict) -> None:
    """
    For hyperparams/config only.
    Uses flor.arg with SAFE names so Flor's CLI kwargs/replay doesn't choke.
    """
    for k, v in cfg_dict.items():
        flor.arg(_safe_arg_name(k), v)

def flor_log_metadata_as_logs(meta_dict: dict) -> None:
    """
    For system/git/observed metadata. These are not "kwargs".
    Use flor.log (more permissive and semantically correct).
    """
    for k, v in meta_dict.items():
        flor.log(k, v)


# ----------------------------
# Main training runner
# ----------------------------

def build_lerobot_train_cmd(cfg: TrainRunConfig) -> list[str]:
    """
    Builds a lerobot training command. Adjust flags to match your actual CLI usage.
    """
    if not cfg.dataset_repo_id:
        raise ValueError("dataset_repo_id is required (HF repo id for LeRobot dataset).")

    cmd = [
        "lerobot-train",
        "--policy.type", cfg.policy_type,
        "--dataset.repo_id", cfg.dataset_repo_id,
        "--output_dir", cfg.output_dir,
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
    

    _SAFE = re.compile(r"[^A-Za-z0-9_]+")

    def safe_arg_key(k: str) -> str:
        k = _SAFE.sub("_", k)
        k = re.sub(r"_+", "_", k)
        return k.strip("_")

    # log config/hparams as args (safe for Flor)
    for k, v in asdict(cfg).items():
        flor.arg(safe_arg_key(f"cfg_{k}"), v)
    # config / hyperparams as args (safe)
    flor_log_config_as_args({f"cfg_{k}": v for k, v in asdict(cfg).items()})

    # system + git metadata as logs (stable columns)
    for k, v in get_static_sys_info().items():
        flor.log(k, v)   # sys/os, git/commit, etc. are stable COLUMNS


    cmd = build_lerobot_train_cmd(cfg)
    flor.arg("train/cmd", " ".join(shlex.quote(x) for x in cmd))

    print("Running:", " ".join(shlex.quote(x) for x in cmd))

    # Start time (stable columns)
    t_start = time.time()
    flor.log("train/run_start_ts", t_start)
    m = try_get_gpu_metrics()
    if m:
        for k, v in m.items():
            flor.log(k, v)
        # optional: also print for console
        print(m)
    else:
        print("GPU metrics unavailable (pynvml not working).")

    # -------------------------
    # Start process
    # -------------------------
    stop_evt = threading.Event()

    def _poll_gpu(stop_evt: threading.Event, every_s: float = 5.0):
        while not stop_evt.is_set():
            m = try_get_gpu_metrics()
            if m:
                for k, v in m.items():
                    flor.log(k, v)
            stop_evt.wait(every_s)

    t = threading.Thread(
        target=_poll_gpu,
        args=(stop_evt, 5.0),   # poll every 5 seconds
        daemon=True,
    )
    t.start()



    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert p.stdout is not None

    # We log repeatedly under a step loop.
    # If the training output includes an explicit step, use it.
    # Otherwise, maintain our own "observed_step" counter (based on metric sightings).
    observed_step: int = 0
    last_logged_step: int = -1

    # For stdout logging (optional)
    line_count = 0

    # Track last seen metrics (so we can log at cadence even if a line is missing some keys)
    last_metrics: Dict[str, Any] = {}

    # For step timing
    last_step_ts = time.time()
    init_nvml(device_index=0)
    print(try_get_gpu_metrics())
    try:
        for raw_line in p.stdout:
            line = raw_line.rstrip("\n")
            print(line)

            line_count += 1

            # Optional stdout logging WITHOUT schema explosion
            if cfg.stdout_every and (line_count % cfg.stdout_every == 0):
                flor.loop("stdout_line", line_count)
                flor.log("train/stdout_line", line)

            # Parse metrics
            m = parse_metrics_from_line(line)
            if m:
                last_metrics.update(m)

            # Determine step
            if "train/step" in last_metrics:
                observed_step = int(last_metrics["train/step"])
            else:
                # If no explicit step appears in logs, do nothing here.
                # You can choose to increment per line or per batch if you have a hook.
                pass

            # Only log on valid steps
            if observed_step <= 0:
                continue

            # Log train metrics every cfg.log_every steps.
            if (observed_step % cfg.log_every == 0) and (observed_step != last_logged_step):
                now = time.time()
                step_dt = now - last_step_ts
                last_step_ts = now
                last_logged_step = observed_step

                flor.loop("step", observed_step)

                # Training metrics (stable column names)
                flor.log("train/step", observed_step)
                flor.log("train/epoch", last_metrics.get("train/epoch", None))
                flor.log("train/loss", last_metrics.get("train/loss", None))
                flor.log("train/lr", last_metrics.get("train/lr", None))
                flor.log("train/step_time_s", float(step_dt))

                # System metrics (stable column names)
                if cfg.sys_every and (observed_step % cfg.sys_every == 0):
                    sysm = get_dynamic_sys_metrics()
                    for k, v in sysm.items():
                        flor.log(k, v)

        rc = p.wait()

        #stop polling
        stop_evt.set()
        t.join(timeout=1.0)

    except KeyboardInterrupt:
        print("\nInterrupted. Terminating training process...")
        try:
            p.terminate()
        except Exception:
            pass
        rc = p.wait()
    finally:
        t_end = time.time()
        flor.log("train/run_end_ts", t_end)
        flor.log("train/run_duration_s", float(t_end - t_start))
        flor.log("train/return_code", int(rc))
        m = try_get_gpu_metrics()
        if m:
            for k, v in m.items():
                flor.log(f"gpu_end/{k.split('/',1)[1] if '/' in k else k}", v)
        shutdown_nvml()

    return int(rc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_type", default="smolvla", help="LeRobot policy type (default: smolvla)")
    ap.add_argument("--dataset_repo_id", required=True, help="HF dataset repo id, e.g. BarbaricErick/my_dataset")
    ap.add_argument("--output_dir", default="outputs/train", help="Training output dir")
    ap.add_argument("--log_every", type=int, default=50, help="Log training metrics every N steps")
    ap.add_argument("--sys_every", type=int, default=50, help="Log system metrics every M steps")
    ap.add_argument("--stdout_every", type=int, default=0, help="Log stdout every K lines (0 disables)")
    ap.add_argument("--run_tag", default="", help="Optional run tag")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (if supported by your training)")
    ap.add_argument("--extra", default=None, help="Extra args passed through to lerobot-train (quoted string)")

    args = ap.parse_args()

    cfg = TrainRunConfig(
        policy_type=args.policy_type,
        dataset_repo_id=args.dataset_repo_id,
        output_dir=args.output_dir,
        log_every=args.log_every,
        sys_every=args.sys_every,
        stdout_every=args.stdout_every,
        run_tag=args.run_tag,
        seed=args.seed,
        extra=args.extra,
    )

    return run_train(cfg)


if __name__ == "__main__":
    raise SystemExit(main())