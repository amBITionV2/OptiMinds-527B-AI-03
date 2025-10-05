# fetch_and_process.py
import time
from .db import get_conn, fetch_images, delete_image_row
from .phone_processor import process_phone_record
from .laptop_processor import process_laptop_record

LOG_TIMING = True

def _fmt_ms(seconds):
    return f"{seconds*1000:.2f}ms"


def fetch_and_process(user_id, task_id):
    t0 = time.perf_counter()
    rows = fetch_images(user_id, task_id)
    t_fetch = time.perf_counter() - t0

    stats = {"fetched": len(rows), "processed": 0, "failed": 0, "deleted": 0}

    if LOG_TIMING:
        print(f"[INFO] Fetched {stats['fetched']} images in {_fmt_ms(t_fetch)}")

    if not rows:
        return stats

    conn = get_conn()
    try:
        for row in rows:
            image_id = row.get("id")
            is_phone = bool(row.get("isPhone"))

            t_start = time.perf_counter()
            try:
                if is_phone:
                    process_phone_record(row)
                else:
                    process_laptop_record(row)

                if image_id is not None:
                    delete_image_row(conn, image_id)
                    stats["deleted"] += 1
                stats["processed"] += 1

                if LOG_TIMING:
                    print(f"  ✔ Done image_id={image_id} isPhone={is_phone} in {_fmt_ms(time.perf_counter()-t_start)}")

            except Exception as e:
                stats["failed"] += 1
                print(f"  ✖ Error image_id={image_id} isPhone={is_phone}: {e}")

    finally:
        conn.close()

    return stats
