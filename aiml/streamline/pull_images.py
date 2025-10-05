# # pull_images.py
# # Fetch & process all rows for (userId, taskId/testId), delete only on success.

# import os
# import time
# from typing import Any, Dict, List

# import psycopg2
# import psycopg2.extras

# # Processors (local, no HTTP)
# from .laptop_processor import process_laptop_record
# from .phone_processor import process_phone_record


# # -----------------------------------------------------------------------------
# # Configuration (override via env vars in production)
# # -----------------------------------------------------------------------------
# PG_DBNAME = os.getenv("PG_DBNAME", "codeeditor1")
# PG_USER = os.getenv("PG_USER", "postgres")
# PG_PASSWORD = os.getenv("PG_PASSWORD", "root1234")
# PG_HOST = os.getenv("PG_HOST", "localhost")
# PG_PORT = os.getenv("PG_PORT", "5432")

# # Which column in table "images" represents the task?
# # Accepts: "testId", "taskId", "test_id", "task_id"
# IMG_TASK_ID_COLUMN = os.getenv("IMG_TASK_ID_COLUMN", "testId")

# # Optional: control verbosity via env
# LOG_TIMING = os.getenv("LOG_TIMING", "1") not in {"0", "false", "False"}


# def _resolve_task_col(env_val: str) -> str:
#     """Map env value to the actual DB column name (snake_case)."""
#     v = (env_val or "").strip().strip('"')
#     if v in {"testId", "test_id"}:
#         return "test_id"
#     if v in {"taskId", "task_id"}:
#         return "task_id"
#     # Default to test_id if unknown
#     return "test_id"


# def get_conn():
#     return psycopg2.connect(
#         dbname=PG_DBNAME,
#         user=PG_USER,
#         password=PG_PASSWORD,
#         host=PG_HOST,
#         port=PG_PORT,
#     )


# def _fmt_ms(seconds: float) -> str:
#     """Format seconds as milliseconds string with 2 decimals."""
#     return f"{seconds * 1000:.2f}ms"


# # -----------------------------------------------------------------------------
# # Read-only fetch (kept for compatibility; now takes task_id)
# # -----------------------------------------------------------------------------
# def fetch_images(user_id: int, task_id: int) -> List[Dict[str, Any]]:
#     """
#     Returns rows with ALL columns (camelCase keys):
#       id, userId, testId, taskId, image_data, time, isPhone

#     NOTE:
#       - DB columns are snake_case (user_id, test_id/task_id, is_phone, time).
#       - We alias them back to camelCase for downstream processors.
#     """
#     task_col_snake = _resolve_task_col(IMG_TASK_ID_COLUMN)

#     # We'll always expose BOTH "testId" and "taskId" in the row dicts for compatibility.
#     # The selected {task_col_snake} value becomes both.
#     sql = f"""
#         SELECT
#             id,
#             user_id          AS "userId",
#             {task_col_snake} AS "testId",
#             {task_col_snake} AS "taskId",
#             image_data,
#             "time",
#             is_phone         AS "isPhone"
#         FROM images
#         WHERE user_id = %s AND {task_col_snake} = %s
#         ORDER BY "time" ASC, id ASC
#     """
#     conn = get_conn()
#     try:
#         with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
#             cur.execute(sql, (user_id, task_id))
#             rows = cur.fetchall()
#             return rows
#     finally:
#         conn.close()


# # -----------------------------------------------------------------------------
# # Fetch → process (by isPhone) → delete on success, with timing logs
# # -----------------------------------------------------------------------------
# def fetch_and_process(user_id: int, task_id: int) -> Dict[str, int]:
#     """
#     Fetch all rows for (userId, taskId/testId), process them based on isPhone,
#     and delete each row from 'images' ONLY IF processing succeeded.

#     Returns counters: {"fetched": N, "processed": P, "failed": F, "deleted": D}
#     Also logs per-image timing: isPhone flag and elapsed time.
#     """
#     t0 = time.perf_counter()
#     rows = fetch_images(user_id=user_id, task_id=task_id)
#     t_fetch = time.perf_counter() - t0

#     stats = {"fetched": len(rows), "processed": 0, "failed": 0, "deleted": 0}
#     if LOG_TIMING:
#         print(
#             f"[INFO] Fetched {stats['fetched']} images for userId={user_id}, taskId={task_id} "
#             f"in {_fmt_ms(t_fetch)}"
#         )

#     if not rows:
#         return stats

#     per_image_times: List[float] = []
#     phone_times: List[float] = []
#     laptop_times: List[float] = []

#     conn = get_conn()
#     try:
#         for i, row in enumerate(rows, start=1):
#             image_id = row["id"]
#             is_phone = bool(row["isPhone"])
#             # Optional: measure image payload size if present
#             try:
#                 img_len = len(bytes(row["image_data"])) if row.get("image_data") is not None else 0
#             except Exception:
#                 img_len = 0

#             if LOG_TIMING:
#                 print(
#                     f"→ Processing [{i}/{len(rows)}] image_id={image_id} "
#                     f"isPhone={is_phone} bytes={img_len}"
#                 )

#             t_start = time.perf_counter()
#             try:
#                 # Route by isPhone
#                 if is_phone:
#                     process_phone_record(row)
#                 else:
#                     process_laptop_record(row)

#                 # Delete only after success
#                 _delete_row(conn, image_id)
#                 stats["processed"] += 1
#                 stats["deleted"] += 1

#                 elapsed = time.perf_counter() - t_start
#                 per_image_times.append(elapsed)
#                 (phone_times if is_phone else laptop_times).append(elapsed)

#                 if LOG_TIMING:
#                     print(
#                         f"  ✔ Done image_id={image_id} "
#                         f"(isPhone={is_phone}) in {_fmt_ms(elapsed)}"
#                     )

#             except Exception as e:
#                 stats["failed"] += 1
#                 elapsed = time.perf_counter() - t_start
#                 per_image_times.append(elapsed)
#                 if is_phone:
#                     phone_times.append(elapsed)
#                 else:
#                     laptop_times.append(elapsed)

#                 print(f"  ✖ Error image_id={image_id} (isPhone={is_phone}) after {_fmt_ms(elapsed)}: {e}")

#         # Summary logs
#         if LOG_TIMING and per_image_times:
#             total_time = sum(per_image_times)
#             avg_all = total_time / len(per_image_times)
#             avg_phone = (sum(phone_times) / len(phone_times)) if phone_times else 0.0
#             avg_laptop = (sum(laptop_times) / len(laptop_times)) if laptop_times else 0.0

#             print(
#                 "[SUMMARY] "
#                 f"fetched={stats['fetched']} processed={stats['processed']} "
#                 f"failed={stats['failed']} deleted={stats['deleted']} "
#                 f"total={_fmt_ms(total_time)} avg_all={_fmt_ms(avg_all)} "
#                 f"avg_phone={_fmt_ms(avg_phone)} avg_laptop={_fmt_ms(avg_laptop)}"
#             )

#         return stats
#     finally:
#         conn.close()


# def _delete_row(conn, image_id: int):
#     """Delete a single row in 'images' by id (commits immediately)."""
#     with conn.cursor() as cur:
#         cur.execute("DELETE FROM images WHERE id = %s", (image_id,))
#         conn.commit()


# # -----------------------------------------------------------------------------
# # Script smoke test (optional)
# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     uid, tid = 1, 1
#     print(f"IMG_TASK_ID_COLUMN={IMG_TASK_ID_COLUMN} -> {_resolve_task_col(IMG_TASK_ID_COLUMN)}")
#     stats = fetch_and_process(uid, tid)
#     print(f"Done: {stats}")

# pull_images.py
# Fetch & process all rows for (userId, taskId/testId), delete only on success.

import os
import time
import json
from typing import Any, Dict, List, Mapping, Optional

import psycopg2
import psycopg2.extras

# Processors (local, no HTTP)
from .laptop_processor import process_laptop_record
from .phone_processor import process_phone_record


# -----------------------------------------------------------------------------
# Configuration (override via env vars in production)
# -----------------------------------------------------------------------------
PG_DBNAME = os.getenv("PG_DBNAME", "codeeditor1")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "root1234")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")

# Which column in table "images" represents the task?
# Accepts: "testId", "taskId", "test_id", "task_id"
IMG_TASK_ID_COLUMN = os.getenv("IMG_TASK_ID_COLUMN", "testId")

# Optional: control verbosity via env
LOG_TIMING = os.getenv("LOG_TIMING", "1") not in {"0", "false", "False"}


def _resolve_task_col(env_val: str) -> str:
    """Map env value to the actual DB column name (snake_case)."""
    v = (env_val or "").strip().strip('"')
    if v in {"testId", "test_id"}:
        return "test_id"
    if v in {"taskId", "task_id"}:
        return "task_id"
    # Default to test_id if unknown
    return "test_id"


def get_conn():
    return psycopg2.connect(
        dbname=PG_DBNAME,
        user=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT,
    )


def _fmt_ms(seconds: float) -> str:
    """Format seconds as milliseconds string with 2 decimals."""
    return f"{seconds * 1000:.2f}ms"


# -----------------------------------------------------------------------------
# Utility: ensure row mapping
# -----------------------------------------------------------------------------
def ensure_row_mapping(row: Any) -> Optional[Mapping]:
    """
    Ensure `row` is a mapping (dict-like). If `row` is a JSON string/bytes, parse it.
    Returns a mapping or None if conversion fails.
    """
    try:
        # RealDictRow and dicts satisfy Mapping
        if isinstance(row, Mapping):
            return row
    except Exception:
        pass

    # If bytes, decode to str
    if isinstance(row, (bytes, bytearray)):
        try:
            row = row.decode("utf-8")
        except Exception:
            return None

    # If string, try JSON parse
    if isinstance(row, str):
        try:
            parsed = json.loads(row)
            if isinstance(parsed, Mapping):
                return parsed
            # If parsed JSON is not an object, wrap it so .get() works.
            return {"value": parsed}
        except Exception:
            return None

    return None


# -----------------------------------------------------------------------------
# Read-only fetch (kept for compatibility; now takes task_id)
# -----------------------------------------------------------------------------
def fetch_images(user_id: int, task_id: int) -> List[Dict[str, Any]]:
    """
    Returns rows with ALL columns (camelCase keys):
      id, userId, testId, taskId, image_data, time, isPhone

    NOTE:
      - DB columns are snake_case (user_id, test_id/task_id, is_phone, time).
      - We alias them back to camelCase for downstream processors.
    """
    task_col_snake = _resolve_task_col(IMG_TASK_ID_COLUMN)

    # We'll always expose BOTH "testId" and "taskId" in the row dicts for compatibility.
    # The selected {task_col_snake} value becomes both.
    sql = f"""
        SELECT
            id,
            user_id          AS "userId",
            {task_col_snake} AS "testId",
            {task_col_snake} AS "taskId",
            image_data,
            "time",
            is_phone         AS "isPhone"
        FROM images
        WHERE user_id = %s AND {task_col_snake} = %s
        ORDER BY "time" ASC, id ASC
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (user_id, task_id))
            rows = cur.fetchall()
            # RealDictCursor returns RealDictRow objects (mapping-like). Convert to dicts
            # to avoid surprises elsewhere.
            return [dict(r) for r in rows]
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Fetch → process (by isPhone) → delete on success, with timing logs
# -----------------------------------------------------------------------------
def fetch_and_process(user_id: int, task_id: int) -> Dict[str, int]:
    """
    Fetch all rows for (userId, taskId/testId), process them based on isPhone,
    and delete each row from 'images' ONLY IF processing succeeded.

    Returns counters: {"fetched": N, "processed": P, "failed": F, "deleted": D}
    Also logs per-image timing: isPhone flag and elapsed time.
    """
    t0 = time.perf_counter()
    rows = fetch_images(user_id=user_id, task_id=task_id)
    t_fetch = time.perf_counter() - t0

    stats = {"fetched": len(rows), "processed": 0, "failed": 0, "deleted": 0}
    if LOG_TIMING:
        print(
            f"[INFO] Fetched {stats['fetched']} images for userId={user_id}, taskId={task_id} "
            f"in {_fmt_ms(t_fetch)}"
        )

    if not rows:
        return stats

    per_image_times: List[float] = []
    phone_times: List[float] = []
    laptop_times: List[float] = []

    conn = get_conn()
    try:
        for i, row in enumerate(rows, start=1):
            image_id = row.get("id") if isinstance(row, dict) else None
            is_phone = bool(row.get("isPhone")) if isinstance(row, dict) else False

            # Optional: measure image payload size if present (defensive)
            row_map = ensure_row_mapping(row)
            if row_map is None:
                img_len = 0
            else:
                try:
                    img_len = len(bytes(row_map["image_data"])) if row_map.get("image_data") is not None else 0
                except Exception:
                    img_len = 0

            if LOG_TIMING:
                print(
                    f"→ Processing [{i}/{len(rows)}] image_id={image_id} "
                    f"isPhone={is_phone} bytes={img_len}"
                )

            t_start = time.perf_counter()
            try:
                # Route by isPhone (call processors with the original dict representation)
                if is_phone:
                    process_phone_record(row if isinstance(row, dict) else row_map)
                else:
                    process_laptop_record(row if isinstance(row, dict) else row_map)

                # Delete only after success
                if image_id is not None:
                    _delete_row(conn, int(image_id))
                    stats["deleted"] += 1
                stats["processed"] += 1

                elapsed = time.perf_counter() - t_start
                per_image_times.append(elapsed)
                (phone_times if is_phone else laptop_times).append(elapsed)

                if LOG_TIMING:
                    print(
                        f"  ✔ Done image_id={image_id} "
                        f"(isPhone={is_phone}) in {_fmt_ms(elapsed)}"
                    )

            except Exception as e:
                stats["failed"] += 1
                elapsed = time.perf_counter() - t_start
                per_image_times.append(elapsed)
                if is_phone:
                    phone_times.append(elapsed)
                else:
                    laptop_times.append(elapsed)

                # Print a helpful preview so debugging is easier
                preview = repr(row)[:300]
                print(f"  ✖ Error image_id={image_id} (isPhone={is_phone}) after {_fmt_ms(elapsed)}: {e}")
                print(f"     row preview: {preview}")

        # Summary logs
        if LOG_TIMING and per_image_times:
            total_time = sum(per_image_times)
            avg_all = total_time / len(per_image_times)
            avg_phone = (sum(phone_times) / len(phone_times)) if phone_times else 0.0
            avg_laptop = (sum(laptop_times) / len(laptop_times)) if laptop_times else 0.0

            print(
                "[SUMMARY] "
                f"fetched={stats['fetched']} processed={stats['processed']} "
                f"failed={stats['failed']} deleted={stats['deleted']} "
                f"total={_fmt_ms(total_time)} avg_all={_fmt_ms(avg_all)} "
                f"avg_phone={_fmt_ms(avg_phone)} avg_laptop={_fmt_ms(avg_laptop)}"
            )

        return stats
    finally:
        conn.close()


def _delete_row(conn, image_id: int):
    """Delete a single row in 'images' by id (commits immediately)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM images WHERE id = %s", (image_id,))
        conn.commit()


# -----------------------------------------------------------------------------
# Script smoke test (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uid, tid = 1, 1
    print(f"IMG_TASK_ID_COLUMN={IMG_TASK_ID_COLUMN} -> {_resolve_task_col(IMG_TASK_ID_COLUMN)}")
    stats = fetch_and_process(uid, tid)
    print(f"Done: {stats}")
