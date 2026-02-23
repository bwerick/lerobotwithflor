import argparse
import datetime as dt
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _try_import_flor():
    """
    FlorDB is optional at runtime. If installed, we'll use it.
    If not installed, we still run the recording and save a manifest.
    """
    try:
        import flor  # type: ignore
        return flor
    except Exception as e:
        return None


def _run_cmd(cmd_list: list[str], env: dict[str, str] | None = None) -> tuple[int, float, str]:
    """
    Run command, stream output to console, and also capture a copy for logging.
    Returns: (return_code, elapsed_s, combined_output)
    """
    start = time.time()
    proc = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    out_lines: list[str] = []
    assert proc.stdout is not None

    for line in proc.stdout:
        sys.stdout.write(line)
        out_lines.append(line)

    rc = proc.wait()
    elapsed = time.time() - start
    return rc, elapsed, "".join(out_lines)


def build_lerobot_record_command(
    root: str,
    robot_port: str,
    teleop_port: str,
    repo_id: str,
    num_episodes: int,
    episode_time_s: int,
    reset_time_s: int,
    push_to_hub: bool,
    single_task: str,
) -> list[str]:
    # This matches your last working config (including episode_time_s=180, reset_time_s=20)
    cameras = {
        "front": {
            "type": "opencv",
            "index_or_path": 0,
            "width": 640,
            "height": 480,
            "fps": 20,
            "fourcc": "MJPG",
        },
        "overhead": {
            "type": "opencv",
            "index_or_path": 1,
            "width": 640,
            "height": 480,
            "fps": 20,
            "fourcc": "MJPG",
        },
        "wrist": {
            "type": "opencv",
            "index_or_path": 2,
            "width": 480,
            "height": 640,
            "fps": 10,
            "fourcc": "MJPG",
            "rotation": 90,
            "warmup_s": 2,
        },
    }

    # lerobot-record expects the camera dict as a JSON-ish string; safest is compact JSON
    cameras_str = json.dumps(cameras, separators=(",", ":"))

    # Note: pass args as a list, not a single string, to avoid PowerShell quoting issues.
    cmd = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={robot_port}",
        "--robot.id=follower",
        f"--robot.cameras={cameras_str}",
        "--teleop.type=so101_leader",
        f"--teleop.port={teleop_port}",
        "--teleop.id=leader",
        "--display_data=true",
        "--display_compressed_images=true",
        f"--dataset.root={root}",
        f'--dataset.repo_id={repo_id}',
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s={reset_time_s}",
        f"--dataset.push_to_hub={'true' if push_to_hub else 'false'}",
        f'--dataset.single_task={single_task}',
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run lerobot-record with FlorDB logging + manifest output.")
    parser.add_argument("--robot_port", default="COM6")
    parser.add_argument("--teleop_port", default="COM5")
    parser.add_argument("--repo_id", default="BarbaricErick/so101_3cam_demo")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--episode_time_s", type=int, default=180)  # up to 3 minutes
    parser.add_argument("--reset_time_s", type=int, default=20)     # 20 seconds reset
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument(
        "--task",
        default="Pick up the red cylinder and place it centered on the white base. Then place the metal stand upright on top of the cylinder.",
    )
    parser.add_argument(
        "--datasets_dir",
        default=str(Path(".") / "datasets"),
        help="Directory where datasets are created (default: ./datasets)",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional run folder name; default creates so101_3cam_YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the command and exit without running.",
    )
    args = parser.parse_args()

    run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"so101_3cam_{run_stamp}"
    datasets_dir = Path(args.datasets_dir).resolve()
    root = str(datasets_dir / run_name)

    # Ensure datasets directory exists
    _safe_mkdir(datasets_dir)

    cmd = build_lerobot_record_command(
        root=root,
        robot_port=args.robot_port,
        teleop_port=args.teleop_port,
        repo_id=args.repo_id,
        num_episodes=args.num_episodes,
        episode_time_s=args.episode_time_s,
        reset_time_s=args.reset_time_s,
        push_to_hub=args.push_to_hub,
        single_task=args.task,
    )

    if args.dry_run:
        print("DRY RUN command:")
        print(" ".join(shlex.quote(x) for x in cmd))
        return

    flor = _try_import_flor()

    # Basic run metadata (also saved to manifest)
    manifest = {
        "tool": "lerobot-record",
        "started_utc": _utc_now_iso(),
        "run_name": run_name,
        "dataset_root": root,
        "host": {
            "platform": platform.platform(),
            "python": sys.version.replace("\n", " "),
            "cwd": str(Path.cwd()),
        },
        "config": {
            "robot_port": args.robot_port,
            "teleop_port": args.teleop_port,
            "repo_id": args.repo_id,
            "num_episodes": args.num_episodes,
            "episode_time_s": args.episode_time_s,
            "reset_time_s": args.reset_time_s,
            "push_to_hub": bool(args.push_to_hub),
            "task": args.task,
        },
        "command": cmd,
    }

    # FlorDB logging: args (run conditions) + key events
    if flor is not None:
        # Record “run-defining” configuration as flor.args
        flor.arg("run_name", run_name)
        flor.arg("dataset_root", root)
        flor.arg("robot_port", args.robot_port)
        flor.arg("teleop_port", args.teleop_port)
        flor.arg("repo_id", args.repo_id)
        flor.arg("num_episodes", args.num_episodes)
        flor.arg("episode_time_s", args.episode_time_s)
        flor.arg("reset_time_s", args.reset_time_s)
        flor.arg("push_to_hub", bool(args.push_to_hub))
        flor.arg("task", args.task)

        # Useful “events” as logs
        flor.log("event", "recording_start")
        flor.log("started_utc", manifest["started_utc"])

    # Run the command
    rc, elapsed_s, combined_output = _run_cmd(cmd)

    manifest["finished_utc"] = _utc_now_iso()
    manifest["return_code"] = rc
    manifest["elapsed_s"] = elapsed_s

    # Keep output small in manifest; still very useful for debugging.
    tail_lines = combined_output.splitlines()[-200:]
    manifest["console_tail"] = "\n".join(tail_lines)

    # Write manifest next to dataset AND also to automation/manifests for easy indexing.
    dataset_manifest_path = Path(root) / "run_manifest.json"
    _safe_mkdir(dataset_manifest_path.parent)
    dataset_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    automation_manifest_dir = Path(__file__).resolve().parent / "manifests"
    _safe_mkdir(automation_manifest_dir)
    (automation_manifest_dir / f"{run_name}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # FlorDB logs on completion
    if flor is not None:
        flor.log("event", "recording_end")
        flor.log("finished_utc", manifest["finished_utc"])
        flor.log("return_code", rc)
        flor.log("elapsed_s", elapsed_s)

        # Track failures as a searchable boolean
        flor.log("success", rc == 0)

        # Optional: log some “known bad signs” from output
        # (useful when diagnosing camera timeouts / dropped frames)
        flor.log("output_contains_timeout", ("Timed out waiting for frame" in combined_output))
        flor.log("output_contains_read_failed", ("read failed (status=False)" in combined_output))

    if rc != 0:
        print("\n---\nRecording FAILED.")
        print(f"Return code: {rc}")
        print(f"Manifest: {dataset_manifest_path}")
        sys.exit(rc)

    print("\n---\nRecording finished OK.")
    print(f"Dataset root: {root}")
    print(f"Manifest: {dataset_manifest_path}")


if __name__ == "__main__":
    main()