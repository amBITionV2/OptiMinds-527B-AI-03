# # # laptop_processor.py
# # # Runs the "isPhone = 0" pipeline locally on an image row and saves the result.

# # import os
# # import json
# # from datetime import datetime, timezone
# # from typing import Dict, Any

# # import psycopg2
# # import psycopg2.extras
# # import numpy as np
# # import cv2

# # # --- your detectors ---
# # from backend.face_verification import verify_identity
# # from backend.face_detection import detect_multiple_faces, detect_faces
# # from backend.device_detection import detect_device
# # from backend.gaze_tracking import detect_gaze
# # from backend.truehuman import process_image as liveness_process_image



# # # -----------------------------
# # # DB config (override with env)
# # # -----------------------------
# # PG_DBNAME = os.getenv("PG_DBNAME", "codeeditor")
# # PG_USER = os.getenv("PG_USER", "postgres")
# # PG_PASSWORD = os.getenv("PG_PASSWORD", "root1234")
# # PG_HOST = os.getenv("PG_HOST", "localhost")
# # PG_PORT = os.getenv("PG_PORT", "5432")


# # def _get_conn():
# #     return psycopg2.connect(
# #         dbname=PG_DBNAME,
# #         user=PG_USER,
# #         password=PG_PASSWORD,
# #         host=PG_HOST,
# #         port=PG_PORT,
# #     )


# # def _ensure_results_table(conn):
# #     """Create results table if it doesn't exist."""
# #     with conn.cursor() as cur:
# #         cur.execute("""
# #             CREATE TABLE IF NOT EXISTS laptop_process_result (
# #                 id            BIGSERIAL PRIMARY KEY,
# #                 image_id      BIGINT    NOT NULL,
# #                 user_id       BIGINT    NOT NULL,
# #                 test_id       BIGINT    NOT NULL,
# #                 time          BIGINT    NOT NULL,
# #                 is_phone      BOOLEAN   NOT NULL,
# #                 processed_at  TIMESTAMPTZ NOT NULL,
# #                 result        JSONB     NOT NULL
# #             );
# #         """)
# #         cur.execute("""
# #             CREATE INDEX IF NOT EXISTS idx_laptop_result_user_test
# #             ON laptop_process_result (user_id, test_id);
# #         """)
# #         conn.commit()


# # def _decode_image(image_bytes: bytes):
# #     """Return OpenCV image (np.ndarray) or None."""
# #     nparr = np.frombuffer(image_bytes, np.uint8)
# #     return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# # def _insert_result(conn, image_id: int, user_id: int, test_id: int,
# #                    time_val: int, is_phone: bool, result: Dict[str, Any]):
# #     with conn.cursor() as cur:
# #         cur.execute("""
# #             INSERT INTO laptop_process_result
# #                 (image_id, user_id, test_id, time, is_phone, processed_at, result)
# #             VALUES
# #                 (%s,       %s,      %s,      %s,   %s,       %s,           %s)
# #         """, (
# #             image_id,
# #             user_id,
# #             test_id,
# #             time_val,
# #             is_phone,
# #             datetime.now(timezone.utc),
# #             json.dumps(result),
# #         ))
# #         conn.commit()


# # def process_laptop_image(
# #     *,
# #     image_id: int,
# #     user_id: int,
# #     test_id: int,
# #     image_data: bytes,
# #     time_val: int,
# #     is_phone: bool,
# # ) -> Dict[str, Any]:
# #     """
# #     Core pipeline for 'isPhone = 0' images.
# #     Takes the 5 inputs pulled from DB, runs detectors locally, saves into Postgres, and returns the result dict.

# #     Returns:
# #         dict result (same structure that your Flask version returned, plus metadata)
# #     Raises:
# #         ValueError if the image cannot be decoded
# #         psycopg2.Error for DB issues
# #     """
# #     # 1) decode
# #     frame = _decode_image(image_data)
# #     if frame is None:
# #         raise ValueError("Invalid image bytes: cv2.imdecode failed")

# #     # 2) run detectors
# #     face_result = detect_faces(frame)
# #     gaze_result = detect_gaze(frame)
# #     device_result = detect_device(frame)
# #     multiple_faces_result = detect_multiple_faces(frame)
# #     # original route passed userId as string; keep parity just in case
# #     identity_result = verify_identity(frame, str(user_id))
# #     liveness_result = liveness_process_image(frame)

# #     result_payload: Dict[str, Any] = {
# #         "face_detection": face_result,
# #         "gaze_detection": gaze_result,
# #         "device_detection": device_result,
# #         "multiple_faces": multiple_faces_result,
# #         "identity_verification": identity_result,
# #         "liveness_detection": liveness_result,
# #         "meta": {
# #             "image_id": image_id,
# #             "user_id": user_id,
# #             "test_id": test_id,
# #             "time": time_val,
# #             "is_phone": is_phone,
# #         }
# #     }

# #     # 3) save to Postgres
# #     conn = _get_conn()
# #     try:
# #         _ensure_results_table(conn)
# #         _insert_result(conn, image_id, user_id, test_id, time_val, is_phone, result_payload)
# #     finally:
# #         conn.close()

# #     return result_payload


# # def process_laptop_record(row: Dict[str, Any]) -> Dict[str, Any]:
# #     """
# #     Convenience wrapper to accept a DB row (from pull_images.fetch_images).
# #     Expected keys: id, userId, testId, image_data, time, isPhone
# #     """
# #     return process_laptop_image(
# #         image_id=int(row["id"]),
# #         user_id=int(row["userId"]),
# #         test_id=int(row["testId"]),
# #         image_data=bytes(row["image_data"]),
# #         time_val=int(row["time"]),
# #         is_phone=bool(row["isPhone"]),
# #     )

# # laptop_processor.py
# # Runs the "isPhone = 0" pipeline locally on an image row and saves the result (including the image).

# import os
# import json
# from datetime import datetime, timezone
# from typing import Dict, Any

# import psycopg2
# import psycopg2.extras
# import numpy as np
# import cv2

# # --- your detectors ---
# from backend.face_verification import verify_identity
# from backend.face_detection import detect_multiple_faces, detect_faces
# from backend.device_detection import detect_device
# from backend.gaze_tracking import detect_gaze
# from backend.truehuman import process_image as liveness_process_image



# # -----------------------------
# # DB config (override with env)
# # -----------------------------
# PG_DBNAME = os.getenv("PG_DBNAME", "codeeditor1")
# PG_USER = os.getenv("PG_USER", "postgres")
# PG_PASSWORD = os.getenv("PG_PASSWORD", "root1234")
# PG_HOST = os.getenv("PG_HOST", "localhost")
# PG_PORT = os.getenv("PG_PORT", "5432")


# def _get_conn():
#     return psycopg2.connect(
#         dbname=PG_DBNAME,
#         user=PG_USER,
#         password=PG_PASSWORD,
#         host=PG_HOST,
#         port=PG_PORT,
#     )


# def _ensure_results_table(conn):
#     """
#     Create results table if it doesn't exist. Also make it forward-compatible:
#     - ensure an `image BYTEA` column exists
#     - drop `image_id` if it exists (we now store the image itself)
#     """
#     with conn.cursor() as cur:
#         # Base table (without image_id)
#         cur.execute("""
#             CREATE TABLE IF NOT EXISTS laptop_process_result (
#                 id            BIGSERIAL PRIMARY KEY,
#                 user_id       BIGINT       NOT NULL,
#                 test_id       BIGINT       NOT NULL,
#                 time          BIGINT       NOT NULL,
#                 is_phone      BOOLEAN      NOT NULL,
#                 processed_at  TIMESTAMPTZ  NOT NULL,
#                 image         BYTEA        NOT NULL,
#                 result        JSONB        NOT NULL
#             );
#         """)
#         # If migrating from the old schema, add image column if missing
#         cur.execute("""ALTER TABLE laptop_process_result
#                        ADD COLUMN IF NOT EXISTS image BYTEA;""")
#         # Make sure image is not null going forward (can only be enforced once data backfilled)
#         # We set NOT NULL here; if you have legacy rows, remove this line or backfill first.
#         try:
#             cur.execute("""ALTER TABLE laptop_process_result
#                            ALTER COLUMN image SET NOT NULL;""")
#         except psycopg2.Error:
#             # If legacy rows prevent NOT NULL, skip hard enforcement
#             conn.rollback()
#         # Drop legacy image_id if present
#         cur.execute("""ALTER TABLE laptop_process_result
#                        DROP COLUMN IF EXISTS image_id;""")
#         # Helpful composite index
#         cur.execute("""
#             CREATE INDEX IF NOT EXISTS idx_laptop_result_user_test
#             ON laptop_process_result (user_id, test_id);
#         """)
#         conn.commit()


# def _decode_image(image_bytes: bytes):
#     """Return OpenCV image (np.ndarray) or None."""
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# def _insert_result(conn, *, user_id: int, test_id: int,
#                    time_val: int, is_phone: bool, image_bytes: bytes,
#                    result: Dict[str, Any]):
#     with conn.cursor() as cur:
#         cur.execute("""
#             INSERT INTO laptop_process_result
#                 (user_id, test_id, time, is_phone, processed_at, image, result)
#             VALUES
#                 (%s,      %s,     %s,   %s,       %s,           %s,    %s)
#         """, (
#             user_id,
#             test_id,
#             time_val,
#             is_phone,
#             datetime.now(timezone.utc),
#             psycopg2.Binary(image_bytes),
#             json.dumps(result),
#         ))
#         conn.commit()


# def process_laptop_image(
#     *,
#     user_id: int,
#     test_id: int,
#     image_data: bytes,
#     time_val: int,
#     is_phone: bool,
# ) -> Dict[str, Any]:
#     """
#     Core pipeline for 'isPhone = 0' images.
#     Takes inputs pulled from DB, runs detectors locally, saves into Postgres (including the image), and returns the result dict.

#     Returns:
#         dict result (same structure your Flask version returned, plus metadata)
#     Raises:
#         ValueError if the image cannot be decoded
#         psycopg2.Error for DB issues
#     """
#     # 1) decode
#     frame = _decode_image(image_data)
#     if frame is None:
#         raise ValueError("Invalid image bytes: cv2.imdecode failed")

#     # 2) run detectors
#     face_result = detect_faces(frame)
#     gaze_result = detect_gaze(frame)
#     device_result = detect_device(frame)
#     multiple_faces_result = detect_multiple_faces(frame)
#     # original route passed userId as string; keep parity just in case
#     identity_result = verify_identity(frame, str(user_id))
#     liveness_result = liveness_process_image(frame)

#     result_payload: Dict[str, Any] = {
#         "face_detection": face_result,
#         "gaze_detection": gaze_result,
#         "device_detection": device_result,
#         "multiple_faces": multiple_faces_result,
#         "identity_verification": identity_result,
#         "liveness_detection": liveness_result,
#         "meta": {
#             "user_id": user_id,
#             "test_id": test_id,
#             "time": time_val,
#             "is_phone": is_phone,
#         }
#     }

#     # 3) save to Postgres (with the image itself)
#     conn = _get_conn()
#     try:
#         _ensure_results_table(conn)
#         _insert_result(
#             conn,
#             user_id=user_id,
#             test_id=test_id,
#             time_val=time_val,
#             is_phone=is_phone,
#             image_bytes=image_data,
#             result=result_payload
#         )
#     finally:
#         conn.close()

#     return result_payload


# def process_laptop_record(row: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Convenience wrapper to accept a DB row (from pull_images.fetch_images).
#     Expected keys: userId, testId, image_data, time, isPhone
#     """
#     return process_laptop_image(
#         user_id=int(row["userId"]),
#         test_id=int(row["testId"]),
#         image_data=bytes(row["image_data"]),
#         time_val=int(row["time"]),
#         is_phone=bool(row["isPhone"]),
#     )

# laptop_processor.py
# Runs the "isPhone = 0" pipeline locally on an image row and saves the result (including the image).
# Detectors are executed concurrently and we wait until all complete.

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Union, Optional
from contextlib import contextmanager
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import psycopg2
import psycopg2.extras
import numpy as np
import cv2

# --- your detectors ---
from backend.face_verification import verify_identity
from backend.face_detection import detect_multiple_faces, detect_faces
from backend.device_detection import detect_device
from backend.gaze_tracking import detect_gaze
from backend.truehuman import process_image as liveness_process_image


# -----------------------------
# DB config (override with env)
# -----------------------------
PG_DBNAME = os.getenv("PG_DBNAME", "codeeditor1")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "root1234")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")

# Use process pool if set (workers must be picklable). Default: threads.
_USE_PROCESS_POOL = os.getenv("LAPTOP_USE_PROCESS_POOL", "0") in {"1", "true", "True"}


def _get_conn():
    return psycopg2.connect(
        dbname=PG_DBNAME,
        user=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT,
    )


def _print_timing(name: str, start: float, end: Optional[float] = None):
    if end is None:
        end = time.perf_counter()
    ms = (end - start) * 1000.0
    print(f"{name}+{ms:.2f}ms (end-st)")


@contextmanager
def _time_block(name: str):
    _start = time.perf_counter()
    try:
        yield
    finally:
        _print_timing(name, _start, time.perf_counter())


def _ensure_results_table(conn):
    """
    Create results table if it doesn't exist. Also make it forward-compatible:
    - ensure an `image BYTEA` column exists
    - drop `image_id` if it exists (we now store the image itself)
    """
    with _time_block("ensure_results_table"):
        with conn.cursor() as cur:
            # Base table (without image_id)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS laptop_process_result (
                    id            BIGSERIAL PRIMARY KEY,
                    user_id       BIGINT       NOT NULL,
                    test_id       BIGINT       NOT NULL,
                    time          BIGINT       NOT NULL,
                    is_phone      BOOLEAN      NOT NULL,
                    processed_at  TIMESTAMPTZ  NOT NULL,
                    image         BYTEA        NOT NULL,
                    result        JSONB        NOT NULL
                );
            """)
            # If migrating from the old schema, add image column if missing
            cur.execute("""ALTER TABLE laptop_process_result
                           ADD COLUMN IF NOT EXISTS image BYTEA;""")
            # Make sure image is not null going forward (can only be enforced once data backfilled)
            try:
                cur.execute("""ALTER TABLE laptop_process_result
                               ALTER COLUMN image SET NOT NULL;""")
            except psycopg2.Error:
                # If legacy rows prevent NOT NULL, skip hard enforcement
                conn.rollback()
            # Drop legacy image_id if present
            cur.execute("""ALTER TABLE laptop_process_result
                           DROP COLUMN IF EXISTS image_id;""")
            # Helpful composite index
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_laptop_result_user_test
                ON laptop_process_result (user_id, test_id);
            """)
            conn.commit()


def _decode_image(image_bytes: bytes):
    """Return OpenCV image (np.ndarray) or None."""
    with _time_block("decode_image"):
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _insert_result(conn, *, user_id: int, test_id: int,
                   time_val: int, is_phone: bool, image_bytes: bytes,
                   result: Union[Dict[str, Any], str]):
    """
    Insert result into laptop_process_result. Normalize result to a JSON object to avoid
    downstream consumers seeing a top-level string.
    """
    with _time_block("insert_result"):
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                result_obj = parsed if isinstance(parsed, dict) else {"value": parsed}
            except Exception:
                result_obj = {"raw": result}
        elif isinstance(result, dict):
            result_obj = result
        else:
            result_obj = {"value": result}

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO laptop_process_result
                    (user_id, test_id, time, is_phone, processed_at, image, result)
                VALUES
                    (%s,      %s,      %s,   %s,       %s,           %s,    %s)
            """, (
                user_id,
                test_id,
                time_val,
                is_phone,
                datetime.now(timezone.utc),
                psycopg2.Binary(image_bytes),
                json.dumps(result_obj),
            ))
            conn.commit()


# -------------------------------------------------------------------------
# Parallel detector runner for laptop pipeline
# -------------------------------------------------------------------------
def _call_with_timing(name: str, fn, *args, **kwargs):
    start = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        _print_timing(name, start)


def _run_detectors_parallel(frame, user_id: int) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Run the set of laptop detectors concurrently and return:
      (face_result, gaze_result, device_result, multiple_faces_result, identity_result, liveness_result)

    By default uses threads (fast and keeps models loaded). If LAPTOP_USE_PROCESS_POOL=1 is set,
    uses a ProcessPoolExecutor (workers must be picklable and will likely reload models).
    """
    # choose executor
    Executor = ProcessPoolExecutor if _USE_PROCESS_POOL else ThreadPoolExecutor

    # We'll run these six functions in parallel:
    # detect_faces, detect_gaze, detect_device, detect_multiple_faces,
    # verify_identity (needs user_id), liveness_process_image
    with Executor(max_workers=6) as ex:
        fut_face = ex.submit(_call_with_timing, "detect_faces", detect_faces, frame)
        fut_gaze = ex.submit(_call_with_timing, "detect_gaze", detect_gaze, frame)
        fut_device = ex.submit(_call_with_timing, "detect_device", detect_device, frame)
        fut_multiple = ex.submit(_call_with_timing, "detect_multiple_faces", detect_multiple_faces, frame)
        # verify_identity requires user_id; pass as string as in original code
        fut_identity = ex.submit(_call_with_timing, "verify_identity", verify_identity, frame, str(user_id))
        fut_liveness = ex.submit(_call_with_timing, "liveness_process_image", liveness_process_image, frame)

        # Wait for all to finish and re-raise any errors
        face_result = fut_face.result()
        gaze_result = fut_gaze.result()
        device_result = fut_device.result()
        multiple_faces_result = fut_multiple.result()
        identity_result = fut_identity.result()
        liveness_result = fut_liveness.result()

    # debug prints of types
    print(f"[DEBUG] detect_faces -> type={type(face_result).__name__}")
    print(f"[DEBUG] detect_gaze -> type={type(gaze_result).__name__}")
    print(f"[DEBUG] detect_device -> type={type(device_result).__name__}")
    print(f"[DEBUG] detect_multiple_faces -> type={type(multiple_faces_result).__name__}")
    print(f"[DEBUG] verify_identity -> type={type(identity_result).__name__}")
    print(f"[DEBUG] liveness_process_image -> type={type(liveness_result).__name__}")

    return face_result, gaze_result, device_result, multiple_faces_result, identity_result, liveness_result


def process_laptop_image(
    *,
    user_id: int,
    test_id: int,
    image_data: bytes,
    time_val: int,
    is_phone: bool,
) -> Dict[str, Any]:
    """
    Core pipeline for 'isPhone = 0' images.
    Takes inputs pulled from DB, runs detectors locally (concurrently), saves into Postgres (including the image), and returns the result dict.
    """
    total_start = time.perf_counter()

    frame = _decode_image(image_data)
    if frame is None:
        raise ValueError("Invalid image bytes: cv2.imdecode failed")

    # Run detectors concurrently and wait for all to finish.
    face_result, gaze_result, device_result, multiple_faces_result, identity_result, liveness_result = \
        _run_detectors_parallel(frame, user_id)

    result_payload: Dict[str, Any] = {
        "face_detection": face_result,
        "gaze_detection": gaze_result,
        "device_detection": device_result,
        "multiple_faces": multiple_faces_result,
        "identity_verification": identity_result,
        "liveness_detection": liveness_result,
        "meta": {
            "user_id": user_id,
            "test_id": test_id,
            "time": time_val,
            "is_phone": is_phone,
        }
    }

    with _time_block("db_connection_open"):
        conn = _get_conn()

    try:
        _ensure_results_table(conn)
        try:
            _insert_result(
                conn,
                user_id=user_id,
                test_id=test_id,
                time_val=time_val,
                is_phone=is_phone,
                image_bytes=image_data,
                result=result_payload
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[ERROR] Exception while inserting laptop result: {e!r}\nTraceback:\n{tb}")
            raise
    finally:
        with _time_block("db_connection_close"):
            conn.close()

    _print_timing("total_pipeline", total_start)
    return result_payload


def process_laptop_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper to accept a DB row (from pull_images.fetch_images).
    Expected keys: userId, testId, image_data, time, isPhone
    """
    # preserve behavior: accept mapping-like row; raise if not
    if not isinstance(row, dict):
        try:
            row = json.loads(row)
        except Exception:
            raise ValueError("process_laptop_record expected dict-like row")

    return process_laptop_image(
        user_id=int(row["userId"]),
        test_id=int(row["testId"]),
        image_data=bytes(row["image_data"]),
        time_val=int(row["time"]),
        is_phone=bool(row["isPhone"]),
    )

