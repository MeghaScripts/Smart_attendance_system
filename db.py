"""
MySQL access layer for the Face Recognition Module.

Uses mysql-connector-python's built-in connection pool. All query helpers
return plain dicts (or lists of dicts) so the rest of the module never has
to deal with cursors directly.
"""
import json
import mysql.connector
import mysql.connector.pooling
from contextlib import contextmanager

from core.config import face_settings

_pool: mysql.connector.pooling.MySQLConnectionPool = None


def init_db_pool():
    """Call once at app startup."""
    global _pool
    if _pool is not None:
        return
    _pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name="face_module_pool",
        pool_size=face_settings.MYSQL_POOL_SIZE,
        host=face_settings.MYSQL_HOST,
        port=face_settings.MYSQL_PORT,
        user=face_settings.MYSQL_USER,
        password=face_settings.MYSQL_PASSWORD,
        database=face_settings.MYSQL_DATABASE,
        autocommit=True,
    )
    print(f"✅ MySQL pool ready ({face_settings.MYSQL_HOST}:{face_settings.MYSQL_PORT}/{face_settings.MYSQL_DATABASE})")


def close_db_pool():
    # mysql-connector pools don't need an explicit close; connections are
    # released back to the pool by each contextmanager below. Nothing to do,
    # kept as a symmetrical shutdown hook for main.py.
    pass


@contextmanager
def get_conn():
    if _pool is None:
        raise RuntimeError("DB pool not initialized — call init_db_pool() at startup")
    conn = _pool.get_connection()
    try:
        yield conn
    finally:
        conn.close()  # returns the connection to the pool


def fetch_one(query: str, params: tuple = ()) -> dict | None:
    with get_conn() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(query, params)
        row = cur.fetchone()
        cur.fetchall()  # drain any unread rows
        cur.close()
        return row


def fetch_all(query: str, params: tuple = ()) -> list[dict]:
    with get_conn() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        return rows


def execute(query: str, params: tuple = ()) -> int:
    """Run INSERT/UPDATE/DELETE. Returns lastrowid (for INSERT) or rowcount."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(query, params)
        result = cur.lastrowid or cur.rowcount
        cur.close()
        return result


def to_json(value) -> str:
    return json.dumps(value)


def from_json(value):
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value  # connector already decoded it
    return json.loads(value)
