import re
import subprocess
import sys
import time

import cv2


def get_camera_backends():
    if sys.platform == "darwin":
        return [
            (cv2.CAP_AVFOUNDATION, "AVFoundation"),
            (cv2.CAP_ANY, "Auto"),
        ]
    if sys.platform.startswith("win"):
        return [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Auto"),
        ]
    return [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY, "Auto"),
    ]


def open_camera(index):
    for backend_id, backend_name in get_camera_backends():
        cap = cv2.VideoCapture(index, backend_id)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for _ in range(5):
            ok, _ = cap.read()
            if ok:
                return cap, backend_name
            time.sleep(0.05)

        cap.release()

    return None, None


def measure_camera_input_fps(cap, sample_seconds=1.2, max_frames=90):
    """
    Estimate the actual incoming camera FPS from blocking read intervals.
    CAP_PROP_FPS is often unreliable for capture devices such as BMD/Blackmagic.
    """
    timestamps = []
    deadline = time.perf_counter() + float(sample_seconds)

    while len(timestamps) < int(max_frames) and time.perf_counter() < deadline:
        ok, _ = cap.read()
        if not ok:
            break
        timestamps.append(time.perf_counter())

    if len(timestamps) < 3:
        return 0.0

    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return 0.0

    return (len(timestamps) - 1) / elapsed


def get_windows_camera_names():
    if not sys.platform.startswith("win"):
        return []

    command = (
        "Get-CimInstance Win32_PnPEntity | "
        "Where-Object { "
        "$_.PNPClass -eq 'Camera' -or "
        "($_.PNPClass -eq 'Image' -and $_.Name -match 'camera|webcam|capture|video|BMD|Blackmagic') "
        "} | Select-Object -ExpandProperty Name"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []

    if result.returncode != 0:
        return []

    names = []
    seen = set()
    for line in result.stdout.splitlines():
        name = line.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def format_camera_choice(index, device_names):
    if index < len(device_names):
        return f"{device_names[index]} (Kamera {index})"
    if len(device_names) == 1:
        return f"{device_names[0]} (Kamera {index})"
    return f"Kamera {index}"


def parse_camera_index(choice):
    match = re.search(r"\(Kamera\s+(\d+)\)\s*$", choice)
    if match:
        return int(match.group(1))
    match = re.search(r"Kamera\s+(\d+)", choice)
    if match:
        return int(match.group(1))
    raise ValueError(f"Kamera-Index nicht gefunden: {choice}")


def get_available_cameras():
    cameras = []
    device_names = get_windows_camera_names()
    for i in range(10):
        cap, _ = open_camera(i)
        if cap is not None:
            cameras.append(format_camera_choice(i, device_names))
            cap.release()
    return cameras if cameras else ["Keine Kamera gefunden"]

