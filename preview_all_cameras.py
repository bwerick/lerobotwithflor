import time
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Settings (edit these)
# ----------------------------
MAX_INDEX = 12          # scan 0..(MAX_INDEX-1)
BACKEND = cv2.CAP_DSHOW # 700 on your machine; use cv2.CAP_MSMF if needed
WARMUP_FRAMES = 3
TARGET_FPS = 30         # UI refresh target; actual cams may differ
READ_TIMEOUT_S = 0.25   # used for quick “is this camera alive?” probing
# ----------------------------


def try_open_camera(index: int):
    cap = cv2.VideoCapture(index, BACKEND)
    if not cap.isOpened():
        cap.release()
        return None, None

    # quick warmup
    frame = None
    ok = False
    t0 = time.time()

    # Try a handful of reads, but bail quickly if it looks dead.
    # NOTE: OpenCV read() can block forever on some bad virtual cams.
    # We keep the attempt minimal; if your system hangs here, lower MAX_INDEX.
    for _ in range(WARMUP_FRAMES):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        if time.time() - t0 > READ_TIMEOUT_S:
            break

    if not ok or frame is None:
        cap.release()
        return None, None

    h, w = frame.shape[:2]
    return cap, (w, h)


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def main():
    print(f"Scanning cameras 0..{MAX_INDEX-1} using backend={int(BACKEND)} ...")

    caps = []
    infos = []

    for i in range(MAX_INDEX):
        cap, wh = try_open_camera(i)
        if cap is None:
            print(f"  [{i}] closed")
            continue

        w, h = wh
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_str = "?" if (fps is None or fps <= 1e-3 or math.isnan(fps)) else f"{fps:.1f}"
        print(f"  [{i}] OPEN  ({w}x{h} @ {fps_str} fps)")
        caps.append((i, cap))
        infos.append((i, w, h, fps_str))

    if not caps:
        print("No working cameras found. Try:")
        print("  - reduce MAX_INDEX")
        print("  - switch BACKEND to cv2.CAP_MSMF")
        return

    n = len(caps)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Matplotlib UI
    plt.ion()
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    fig.canvas.manager.set_window_title("Camera Grid Preview (press 'q' to quit)")

    # Normalize axes structure
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Create image artists
    artists = []
    for ax_i, ax in enumerate(axes):
        ax.axis("off")
        if ax_i < n:
            cam_index, _, _, fps_str = infos[ax_i]
            ax.set_title(f"cam {cam_index} ({fps_str} fps)", fontsize=10)
            # placeholder image
            placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
            im = ax.imshow(placeholder)
            artists.append(im)
        else:
            ax.set_title("")
            im = ax.imshow(np.zeros((10, 10, 3), dtype=np.uint8))
            artists.append(im)

    quitting = {"flag": False}

    def on_key(event):
        if event.key == "q":
            quitting["flag"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.tight_layout()

    # Main loop
    frame_interval = 1.0 / max(1, TARGET_FPS)
    last = time.time()

    try:
        while not quitting["flag"] and plt.fignum_exists(fig.number):
            now = time.time()
            if now - last < frame_interval:
                time.sleep(0.001)
                continue
            last = now

            for k, (cam_index, cap) in enumerate(caps):
                ok, frame = cap.read()
                if not ok or frame is None:
                    # show a red-ish “dead frame”
                    dead = np.zeros((240, 320, 3), dtype=np.uint8)
                    dead[..., 0] = 120
                    artists[k].set_data(dead)
                    continue

                rgb = bgr_to_rgb(frame)
                artists[k].set_data(rgb)

            fig.canvas.draw_idle()
            plt.pause(0.001)

    finally:
        for _, cap in caps:
            try:
                cap.release()
            except Exception:
                pass
        plt.close(fig)
        print("Closed all cameras.")


if __name__ == "__main__":
    main()