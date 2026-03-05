#!/usr/bin/env python3
"""
automation/train_with_florcl.py

Fine-tune (or train) a LeRobot policy (default: SmolVLA) while logging:
- run config / hyperparams (once) via flor.arg(...)
- training metrics (every N steps) via flor.loop("step", [step]) + flor.log(...)
- system metrics (every M steps) via flor.log(...)

Key FlorDB rules:
  1. flor.log(<name>, <value>) -> <name> becomes a COLUMN in the pivoted view.
     So <name> must be STABLE (e.g., "train/loss"), NOT "train/loss/step_00042".
  2. flor.loop() is a GENERATOR — always use as: for _ in flor.loop("name", iterable):
     Never call it as a standalone statement.
  3. flor.arg() names must be CLI-safe (no slashes, spaces, or special chars).

Cross-platform:
- psutil is optional but recommended for CPU/RAM metrics
- NVIDIA GPU util% is optional (pynvml). CUDA mem stats via torch if available.

Run inside a Git repo (Flor expects that).
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import time
import platform
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import flordb as flor

# --------------------------------------------------------------------------
# Optional dependencies — all wrapped so the script runs without them
# --------------------------------------------------------------------------

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None  # type: ignore


# --------------------------------------------------------------------------
# NVML lifecycle  (module-level; do NOT call nvmlInit/Shutdown elsewhere)
# --------------------------------------------------------------------------

_NVML_READY = False
_NVML_HANDLE = None


def init_nvml(device_index: int = 0) -> None:
    global _NVML_READY, _NVML_HANDLE
    if _NVML_READY:
        return
    if pynvml is None:
        return
    try:
        pynvml.nvmlInit()
        _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        _NVML_READY = True
    except Exception:
        _NVML_READY = False
        _NVML_HANDLE = None


def shutdown_nvml() -> None:
    global _NVML_READY, _NVML_HANDLE
    if not _NVML_READY or pynvml is None:
        return
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
    _NVML_READY = False
    _NVML_HANDLE = None


def try_get_gpu_metrics(prefix: str = "gpu") -> Dict[str, Any]:
    """
    Returns dict with GPU metrics or {} if unavailable.
    Uses only the module-level _NVML_HANDLE — never calls nvmlInit/Shutdown.
    prefix lets you log "gpu/..." vs "gpu_end/..." at run end.
    """
    if not _NVML_READY or _NVML_HANDLE is None or pynvml is None:
        return {}
    try:
        h = _NVML_HANDLE
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")

        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)

        out: Dict[str, Any] = {
            f"{prefix}/name": name,
            f"{prefix}/util_pct": int(util.gpu),
            f"{prefix}/mem_util_pct": int(util.memory),
            f"{prefix}/mem_used_mb": round(mem.used / (1024 ** 2), 1),
            f"{prefix}/mem_total_mb": round(mem.total / (1024 ** 2), 1),
            f"{prefix}/mem_free_mb": round(mem.free / (1024 ** 2), 1),
            f"{prefix}/temp_c": int(temp),
        }
        try:
            mw = pynvml.nvmlDeviceGetPowerUsage(h)
            out[f"{prefix}/power_w"] = round(mw / 1000.0, 1)
        except Exception:
            pass
        try:
            out[f"{prefix}/sm_clock_mhz"] = int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
            out[f"{prefix}/mem_clock_mhz"] = int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
        except Exception:
            pass
        return out
    except Exception:
        return {}


# --------------------------------------------------------------------------
# Git / static system info
# --------------------------------------------------------------------------

def _run_cmd(cmd: list) -> Optional[str]:
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
        "sys/psutil_available": psutil is not None,
        "sys/pynvml_available": pynvml is not None,
    }
    if torch is not None:
        info["sys/torch_version"] = getattr(torch, "__version__", None)
        try:
            cuda_ok = bool(torch.cuda.is_available())
            info["sys/cuda_available"] = cuda_ok
        except Exception:
            cuda_ok = False
            info["sys/cuda_available"] = None
        if cuda_ok:
            try:
                info["sys/cuda_version"] = getattr(torch.version, "cuda", None)
                info["sys/gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                info["sys/cuda_version"] = None
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
    return info


def get_dynamic_sys_metrics() -> Dict[str, Any]:
    """
    Sampled repeatedly at logging cadence.
    NOTE: never calls nvmlInit/Shutdown — relies on module-level lifecycle.
    """
    m: Dict[str, Any] = {}

    # CPU / RAM
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

    # GPU memory via torch
    cuda_ok = False
    if torch is not None:
        try:
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            pass

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

    # GPU util% via module-level pynvml handle (no init/shutdown here)
    gpu_m = try_get_gpu_metrics(prefix="gpu")
    m["sys/gpu_util_pct"] = gpu_m.get("gpu/util_pct", None)

    return m


# --------------------------------------------------------------------------
# LeRobot stdout metric parsing
# --------------------------------------------------------------------------

# Equals-style patterns (generic trainers): loss=1.41, lr=1e-4, step=5
_EQUALS_PATTERNS = [
    (re.compile(r"\b(?:step|global_step)\s*=\s*(\d+)\b", re.IGNORECASE),               "train/step",  int),
    (re.compile(r"\bepoch\s*=\s*(\d+)\b", re.IGNORECASE),                              "train/epoch", float),
    (re.compile(r"\bloss\s*=\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\b"),                "train/loss",  float),
    (re.compile(r"\b(?:lr|learning_rate)\s*=\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\b",
                re.IGNORECASE),                                                          "train/lr",    float),
]

# Colon-style patterns (LeRobot format): step:1 loss:1.410 lr:1.0e-04 epch:0.00
_COLON_PATTERNS = [
    (re.compile(r"\bstep:(\d+)\b"),                                                     "train/step",  int),
    (re.compile(r"\bepch:([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"),                         "train/epoch", float),
    (re.compile(r"\bloss:([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"),                         "train/loss",  float),
    (re.compile(r"\blr:([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"),                           "train/lr",    float),
]


def parse_metrics_from_line(line: str) -> Dict[str, Any]:
    """
    Extract metrics from a single stdout line.
    Handles both LeRobot colon-style (step:1 loss:1.41) and
    generic equals-style (step=1 loss=1.41).
    Colon-style takes priority when both match.
    """
    out: Dict[str, Any] = {}

    # Try colon-style first (LeRobot)
    for pat, key, typ in _COLON_PATTERNS:
        m = pat.search(line)
        if m:
            try:
                out[key] = typ(m.group(1))
            except (ValueError, TypeError):
                pass

    # Fill any missing keys with equals-style
    for pat, key, typ in _EQUALS_PATTERNS:
        if key in out:
            continue  # already found via colon-style
        m = pat.search(line)
        if m:
            try:
                out[key] = typ(m.group(1))
            except (ValueError, TypeError):
                pass

    return out


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

_SAFE_ARG_RE = re.compile(r"[^A-Za-z0-9_]+")


def _safe_arg_key(name: str) -> str:
    """Make a name safe for flor.arg() — CLI-friendly, no slashes or spaces."""
    name = _SAFE_ARG_RE.sub("_", name.strip())
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


@dataclass
class TrainRunConfig:
    policy_type: str = "smolvla"
    dataset_repo_id: str = ""
    output_dir: str = "outputs/train"
    log_every: int = 50
    sys_every: int = 50
    run_tag: str = ""
    seed: int = 0
    extra: Optional[str] = None


# --------------------------------------------------------------------------
# Main training runner
# --------------------------------------------------------------------------

def build_lerobot_train_cmd(cfg: TrainRunConfig) -> list:
    if not cfg.dataset_repo_id:
        raise ValueError("dataset_repo_id is required.")
    cmd = [
        "lerobot-train",
        "--policy.type", cfg.policy_type,
        "--dataset.repo_id", cfg.dataset_repo_id,
        "--output_dir", cfg.output_dir,
    ]
    if cfg.seed:
        cmd += ["--seed", str(cfg.seed)]
    if cfg.extra:
        cmd += shlex.split(cfg.extra)
    return cmd


def run_train(cfg: TrainRunConfig) -> int:
    # -----------------------------------------------------------------------
    # init_nvml ONCE at the top, before anything else touches the GPU
    # -----------------------------------------------------------------------
    init_nvml(device_index=0)

    # -----------------------------------------------------------------------
    # Log config once via flor.arg (CLI-safe names, no slashes)
    # -----------------------------------------------------------------------
    for k, v in asdict(cfg).items():
        flor.arg(_safe_arg_key(f"cfg_{k}"), v)

    # Static system + git info as flor.log (observed metadata, not kwargs)
    for k, v in get_static_sys_info().items():
        flor.log(k, v)

    # train_cmd as arg — note: NO slash in name
    cmd = build_lerobot_train_cmd(cfg)
    flor.arg("train_cmd", " ".join(shlex.quote(x) for x in cmd))

    print("Running:", " ".join(shlex.quote(x) for x in cmd))

    t_start = time.time()
    flor.log("train/run_start_ts", t_start)

    # Log initial GPU state
    for k, v in try_get_gpu_metrics(prefix="gpu").items():
        flor.log(k, v)

    # -----------------------------------------------------------------------
    # Subprocess
    # -----------------------------------------------------------------------
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert p.stdout is not None

    observed_step: int = -1          # -1 means "not yet seen"
    last_logged_step: int = -1
    last_metrics: Dict[str, Any] = {}
    last_step_ts = time.time()
    rc = 1  # default in case of early exception

    try:
        for raw_line in p.stdout:
            line = raw_line.rstrip("\n")
            print(line)

            # Parse any metrics from this line
            parsed = parse_metrics_from_line(line)
            if parsed:
                last_metrics.update(parsed)

            if "train/step" in last_metrics:
                observed_step = int(last_metrics["train/step"])

            # Skip until we've seen at least one real step
            if observed_step < 0:
                continue

            if (observed_step % cfg.log_every == 0) and (observed_step != last_logged_step):
                now = time.time()
                step_dt = now - last_step_ts
                last_step_ts = now
                last_logged_step = observed_step

                # flor.loop() used as a generator with a single-element list
                # to register the step context for this subprocess-driven step
                for _ in flor.loop("step", [observed_step]):
                    flor.log("train/step", observed_step)
                    flor.log("train/epoch", last_metrics.get("train/epoch"))
                    flor.log("train/loss", last_metrics.get("train/loss"))
                    flor.log("train/lr", last_metrics.get("train/lr"))
                    flor.log("train/step_time_s", float(step_dt))

                    if cfg.sys_every and (observed_step % cfg.sys_every == 0):
                        for k, v in get_dynamic_sys_metrics().items():
                            flor.log(k, v)

        rc = p.wait()

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

        # Log final GPU state under "gpu_end/" prefix for easy comparison
        for k, v in try_get_gpu_metrics(prefix="gpu_end").items():
            flor.log(k, v)

        # Always shut down nvml last
        shutdown_nvml()

    return int(rc)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Train a LeRobot policy with FlorDB logging.")
    ap.add_argument("--policy_type", default="smolvla")
    ap.add_argument("--dataset_repo_id", required=True,
                    help="HF dataset repo id, e.g. BarbaricErick/my_dataset")
    ap.add_argument("--output_dir", default="outputs/train")
    ap.add_argument("--log_every", type=int, default=50,
                    help="Log training metrics every N steps")
    ap.add_argument("--sys_every", type=int, default=50,
                    help="Log system metrics every M steps")
    ap.add_argument("--run_tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", default=None,
                    help="Extra args passed through to lerobot-train (quoted string)")
    args = ap.parse_args()

    cfg = TrainRunConfig(
        policy_type=args.policy_type,
        dataset_repo_id=args.dataset_repo_id,
        output_dir=args.output_dir,
        log_every=args.log_every,
        sys_every=args.sys_every,
        run_tag=args.run_tag,
        seed=args.seed,
        extra=args.extra,
    )
    return run_train(cfg)


if __name__ == "__main__":
    raise SystemExit(main())