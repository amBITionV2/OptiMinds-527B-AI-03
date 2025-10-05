# phone_processor.py
import time
import json
import numpy as np
import cv2
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from .db import get_conn, ensure_phone_result_table, insert_phone_result

# Import your existing detectors
from projectroot.detection.person_detection import detect_person
from projectroot.detection.electronic_device_detection import detect_electronic_devices
from projectroot.detection.notebook_detection import detect_notebook
from projectroot.detection.activity_recognition import recognize_activities


def _print_timing(name, start, end=None):
    if end is None:
        end = time.perf_counter()
    print(f"{name}+{(end-start)*1000:.2f}ms")


def _decode_image(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _run_detectors_parallel(frame):
    def _call_with_timing(name, fn, *args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _print_timing(name, start)
    
    with ThreadPoolExecutor(max_workers=4) as ex:
        fut_person = ex.submit(_call_with_timing, "detect_person", detect_person, frame)
        fut_devices = ex.submit(_call_with_timing, "detect_electronic_devices", detect_electronic_devices, frame)
        fut_notebook = ex.submit(_call_with_timing, "detect_notebook", detect_notebook, frame)
        fut_activity = ex.submit(_call_with_timing, "recognize_activities", recognize_activities, frame)

        return (fut_person.result(), fut_devices.result(), fut_notebook.result(), fut_activity.result())


def process_phone_record(row):
    user_id = int(row["userId"])
    test_id = int(row["testId"])
    time_val = int(row["time"])
    is_phone = bool(row["isPhone"])
    image_data = bytes(row["image_data"])

    frame = _decode_image(image_data)
    if frame is None:
        raise ValueError("Invalid image bytes")

    person, devices, notebook, activity = _run_detectors_parallel(frame)

    result_payload = {
        "person": person,
        "electronic_devices": devices,
        "notebook": notebook,
        "activity": activity,
        "meta": {"user_id": user_id, "test_id": test_id, "time": time_val, "is_phone": is_phone}
    }

    conn = get_conn()
    try:
        ensure_phone_result_table(conn)
        insert_phone_result(conn, user_id, test_id, time_val, is_phone, image_data, result_payload)
    finally:
        conn.close()

    return result_payload
