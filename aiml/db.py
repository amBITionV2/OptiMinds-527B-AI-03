# db.py
import os
import mysql.connector
from mysql.connector import Error
import json

MYSQL_DBNAME = os.getenv("MYSQL_DBNAME", "codeeditor1")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root1234")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))


def get_conn():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DBNAME,
        port=MYSQL_PORT
    )


def ensure_phone_result_table(conn):
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS phone_result (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        user_id BIGINT NOT NULL,
        test_id BIGINT NOT NULL,
        time BIGINT NOT NULL,
        is_phone BOOLEAN NOT NULL,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        image LONGBLOB NOT NULL,
        result JSON NOT NULL,
        INDEX idx_phone_result_user_test(user_id, test_id)
    )
    """)
    conn.commit()


def insert_phone_result(conn, user_id, test_id, time_val, is_phone, image_bytes, result_obj):
    cursor = conn.cursor()
    sql = """
        INSERT INTO phone_result
        (user_id, test_id, time, is_phone, processed_at, image, result)
        VALUES (%s,%s,%s,%s,NOW(),%s,%s)
    """
    cursor.execute(sql, (
        user_id,
        test_id,
        time_val,
        is_phone,
        image_bytes,
        json.dumps(result_obj)
    ))
    conn.commit()


def fetch_images(user_id, task_id):
    conn = get_conn()
    cursor = conn.cursor(dictionary=True)
    sql = """
        SELECT id, user_id AS userId, test_id AS testId, task_id AS taskId,
               image_data, time, is_phone AS isPhone
        FROM images
        WHERE user_id=%s AND test_id=%s
        ORDER BY time ASC, id ASC
    """
    cursor.execute(sql, (user_id, task_id))
    rows = cursor.fetchall()
    conn.close()
    return rows


def delete_image_row(conn, image_id):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM images WHERE id=%s", (image_id,))
    conn.commit()
