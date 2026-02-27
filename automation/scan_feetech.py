# scan_feetech.py
from __future__ import annotations

import sys
import time

PORT = "COM5"         # <-- change if needed
BAUD = 1000000        # common for Feetech bus; if no hits, we’ll try others
ID_RANGE = range(1, 21)

def main():
    try:
        import serial
    except Exception as e:
        print("pyserial missing?", e)
        sys.exit(1)

    # Try to import the Feetech SDK that lerobot installed
    try:
        from scservo_sdk import PortHandler, PacketHandler  # Feetech/SCServo SDK
    except Exception as e:
        print("Could not import scservo_sdk (feetech). Did you install the extra?")
        print("pip install -e '.[feetech]'")
        print("Error:", e)
        sys.exit(1)

    port = PortHandler(PORT)
    if not port.openPort():
        print(f"FAILED to open {PORT}. Is something else using it?")
        sys.exit(1)

    if not port.setBaudRate(BAUD):
        print(f"FAILED to set baud {BAUD} on {PORT}")
        sys.exit(1)

    packet = PacketHandler(1)  # protocol version; many Feetech servos use 1
    found = []
    print(f"Scanning {PORT} @ {BAUD} for IDs {ID_RANGE.start}..{ID_RANGE.stop-1} ...")

    for dxl_id in ID_RANGE:
        # ping returns (model_number, comm_result, error)
        model_number, comm_result, error = packet.ping(port, dxl_id)
        if comm_result == 0 and error == 0:
            found.append((dxl_id, model_number))
            print(f"  Found ID {dxl_id} (model {model_number})")
        time.sleep(0.02)

    port.closePort()

    if not found:
        print("No motors found at this baud/protocol.")
        print("Next: try different BAUD values (115200, 250000, 500000, 1000000).")
    else:
        print("\nSummary:", found)

if __name__ == "__main__":
    main()
