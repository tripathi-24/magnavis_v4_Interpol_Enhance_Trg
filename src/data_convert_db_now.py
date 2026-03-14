"""
Database-backed magnetic time-series ingestion for Magnavis.

This module is meant to be a drop-in replacement for `data_convert_now.py`:
- It exposes `get_timeseries_magnetic_data(...)` with a compatible signature.
- It returns a DataFrame with the same column names expected by `application.py`:
  - `time_H`
  - `mag_H_nT`

Data source:
- MySQL table: `qnav_magneticdatamodel`
- Credentials/host are taken from the reference script `Get_Data_09Dec25.py`.

Notes:
- The database schema (as seen in the provided CSV) includes vector components `b_x`, `b_y`, `b_z`.
- `application.py` expects a single scalar magnetic series. Here we compute **total field magnitude**
  (sqrt(b_x^2 + b_y^2 + b_z^2)) and expose it as `mag_H_nT` to keep the rest of the app unchanged.
  If you prefer a different scalar (e.g., `b_x` only or horizontal magnitude), we can switch it.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

try:
    import mysql.connector
except Exception as e:  # pragma: no cover
    mysql = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


APP_BASE = os.path.dirname(__file__)
TABLE_NAME = "qnav_magneticdatamodel"

# Copied from Get_Data_09Dec25.py (reference script provided by user)
DB_CONFIG = {
    "host": "50.63.129.30",
    "user": "devuser",
    "password": "devuser@221133",
    "database": "dbqnaviitk",
    "port": 3306,
    "connection_timeout": 60,
    "read_timeout": 7200,
    "write_timeout": 7200,
    "use_pure": True,
    "buffered": False,
    "autocommit": False,
    "pool_reset_session": True,
}

# Robustness settings (adapted from Get_Data_09Dec25.py)
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2
MAX_BACKOFF_SECONDS = 30
LONG_NET_TIMEOUT_SECONDS = 7200
FETCH_CHUNK_SIZE = 2000  # fetchmany size


def _ensure_mysql_available() -> None:
    if _IMPORT_ERR is not None:
        raise ImportError(
            "mysql.connector is required for DB ingestion. "
            "Install mysql-connector-python. Original error: "
            f"{_IMPORT_ERR}"
        )


def _get_latest_sensor_id(conn) -> Optional[str]:
    """
    Try to pick the most recent sensor_id so we fetch a consistent stream.
    If the query fails, return None and fetch without sensor filter.
    """
    try:
        cur = conn.cursor()
        # Prefer ORDER BY id (primary key) for speed and index usage.
        cur.execute(f"SELECT sensor_id FROM {TABLE_NAME} ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        cur.close()
        if row and row[0]:
            return str(row[0])
    except Exception:
        return None
    return None


def _get_max_timestamp(sensor_id: Optional[str] = None) -> Optional[datetime]:
    """Return the most recent timestamp present in the table (optionally filtered by sensor_id)."""
    _ensure_mysql_available()
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        _set_session_timeouts(conn)
        cur = conn.cursor()
        if sensor_id:
            # Use primary key order to locate the latest row for this sensor quickly.
            cur.execute(
                f"SELECT timestamp FROM {TABLE_NAME} WHERE sensor_id=%s ORDER BY id DESC LIMIT 1",
                (sensor_id,),
            )
        else:
            cur.execute(f"SELECT timestamp FROM {TABLE_NAME} ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        cur.close()
        if row and row[0]:
            # mysql.connector may return datetime already; normalize to python datetime.
            ts = row[0]
            ts = pd.to_datetime(ts).to_pydatetime()
            # keep timezone-naive
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            return ts
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return None


def get_latest_sensor_ids(limit: int = 6) -> list[str]:
    """
    Return up to `limit` distinct sensor_ids, ordered by most recent data (by id DESC).

    This is used by application_temp.py to discover the 6 streams
    (2 observatories × 3 sensors each).
    """
    _ensure_mysql_available()
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        _set_session_timeouts(conn)
        cur = conn.cursor()
        cur.execute(f"SELECT DISTINCT sensor_id FROM {TABLE_NAME} ORDER BY id DESC LIMIT %s", (int(limit),))
        rows = cur.fetchall()
        cur.close()
        out = []
        for r in rows or []:
            if r and r[0]:
                out.append(str(r[0]))
        return out
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_latest_sensor_id_like(pattern: str) -> Optional[str]:
    """
    Return the most recent sensor_id matching a SQL LIKE pattern, e.g. '%OBS1_1'.
    Uses ORDER BY id DESC to pick the latest stream variant.
    """
    _ensure_mysql_available()
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        _set_session_timeouts(conn)
        cur = conn.cursor()
        cur.execute(
            f"SELECT sensor_id FROM {TABLE_NAME} WHERE sensor_id LIKE %s ORDER BY id DESC LIMIT 1",
            (pattern,),
        )
        row = cur.fetchone()
        cur.close()
        if row and row[0]:
            return str(row[0])
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return None


def get_timeseries_magnetic_data_multi(
    sensor_ids: list[str],
    *,
    hours: float = 1.0,
    last_n_samples: int = 3600,  # 60 min @ 1 Hz
) -> dict[str, pd.DataFrame]:
    """
    Fetch the most recent `hours` of data for each sensor_id.

    Returns:
      dict sensor_id -> DataFrame with columns [time_H, mag_H_nT]
    """
    results: dict[str, pd.DataFrame] = {}
    for sid in sensor_ids:
        try:
            end_ts = _get_max_timestamp(sensor_id=sid)
            if end_ts is None:
                results[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
                continue
            start_ts = end_ts - timedelta(hours=float(hours))
            # We may have multiple samples per second; fetch more raw rows and then
            # downsample to 1 Hz (one averaged value per second).
            raw_limit_rows = max(int(last_n_samples) * 3, 12000)
            df_raw = _query_db(
                start_time=start_ts,
                end_time=end_ts,
                limit_rows=int(raw_limit_rows),
                sensor_id=sid,
                order_desc=True,
            )
            results[sid] = _raw_to_timeseries_df(df_raw, target_n_seconds=int(last_n_samples))
        except Exception:
            results[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
    return results


def get_timeseries_magnetic_data_since_multi(
    sensor_ids: list[str],
    *,
    since_times: dict[str, datetime],
    limit_rows: int = 5000,
) -> dict[str, pd.DataFrame]:
    """
    Incremental fetch: for each sensor, fetch rows from (since_time .. latest_db_time].

    Returns dict sensor_id -> DataFrame [time_H, mag_H_nT]. If no new data, returns empty DF for that sensor.
    """
    out: dict[str, pd.DataFrame] = {}
    for sid in sensor_ids:
        try:
            since = since_times.get(sid)
            if since is None:
                out[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
                continue
            end_ts = _get_max_timestamp(sensor_id=sid)
            if end_ts is None or end_ts <= since:
                out[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
                continue
            df_raw = _query_db(
                start_time=since,
                end_time=end_ts,
                limit_rows=int(limit_rows),
                sensor_id=sid,
                order_desc=False,
            )
            # Incremental: still average within each second; don't hard-cap here.
            out[sid] = _raw_to_timeseries_df(df_raw, target_n_seconds=None)
        except Exception:
            out[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
    return out


def _get_min_timestamp_at_or_after(sensor_id: str, start_time: datetime) -> Optional[datetime]:
    """Find the first available timestamp >= start_time for a given sensor_id."""
    _ensure_mysql_available()
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        _set_session_timeouts(conn)
        cur = conn.cursor()
        cur.execute(
            f"SELECT timestamp FROM {TABLE_NAME} WHERE sensor_id=%s AND timestamp >= %s ORDER BY id ASC LIMIT 1",
            (sensor_id, start_time),
        )
        row = cur.fetchone()
        cur.close()
        if row and row[0]:
            ts = pd.to_datetime(row[0]).to_pydatetime()
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            return ts
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return None


def get_min_timestamp_at_or_after(sensor_id: str, start_time: datetime) -> Optional[datetime]:
    """Public wrapper around `_get_min_timestamp_at_or_after` (used by simulation mode)."""
    return _get_min_timestamp_at_or_after(sensor_id, start_time)


def fetch_timeseries_window_multi(
    sensor_ids: list[str],
    *,
    start_time: datetime,
    end_time: datetime,
    target_n_seconds: int = 3600,  # 60 min @ 1 Hz
) -> dict[str, pd.DataFrame]:
    """
    Fetch a fixed window [start_time, end_time] for each sensor_id and return a 1 Hz series.
    If a sensor has no data in that window, we shift that sensor's start_time forward to the
    first available timestamp >= start_time (sensor-local) and retry once.
    """
    out: dict[str, pd.DataFrame] = {}
    duration = end_time - start_time
    for sid in sensor_ids:
        try:
            # Raw DB can have many samples per second (sub-second sampling).
            # To reliably obtain `target_n_seconds` unique seconds after 1 Hz averaging,
            # we must fetch *far more* than target_n_seconds rows.
            # Use a generous multiplier with a safety cap to avoid huge pulls.
            base_limit = int(target_n_seconds) * 50
            raw_limit_rows = min(max(base_limit, 12000), 500000)

            # Align start per sensor to the first available point >= requested start_time.
            s2 = _get_min_timestamp_at_or_after(sid, start_time) or start_time
            e2 = s2 + duration

            # Expand forward until we accumulate enough 1 Hz seconds (or hit max expansions).
            df_ts = pd.DataFrame(columns=["time_H", "mag_H_nT"])
            max_expands = 12  # up to ~12 hours if duration is 1 hour (conservative)
            expands = 0
            while expands <= max_expands:
                # As we expand the time window, also increase the row limit a bit, because
                # high-rate sensors can still compress time coverage under a fixed LIMIT.
                limit_this = min(int(raw_limit_rows * (1 + 0.25 * expands)), 800000)
                df_raw = _query_db(
                    start_time=s2,
                    end_time=e2,
                    limit_rows=int(limit_this),
                    sensor_id=sid,
                    order_desc=False,
                )
                df_ts = _raw_to_timeseries_df(df_raw, target_n_seconds=None)
                if df_ts is not None and len(df_ts) >= int(target_n_seconds):
                    df_ts = df_ts.tail(int(target_n_seconds)).reset_index(drop=True)
                    break
                # not enough data yet: extend end forward
                e2 = e2 + duration
                expands += 1

            out[sid] = df_ts if df_ts is not None else pd.DataFrame(columns=["time_H", "mag_H_nT"])
        except Exception:
            out[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
    return out


def fetch_timeseries_between_multi(
    sensor_ids: list[str],
    *,
    start_time: datetime,
    end_time: datetime,
    limit_rows: int = 20000,
) -> dict[str, pd.DataFrame]:
    """
    Fetch a forward window (start_time, end_time] for each sensor_id and return a 1 Hz series (no hard cap).
    Intended for simulated "realtime" incremental updates.
    """
    out: dict[str, pd.DataFrame] = {}
    for sid in sensor_ids:
        try:
            df_raw = _query_db(
                start_time=start_time,
                end_time=end_time,
                limit_rows=int(limit_rows),
                sensor_id=sid,
                order_desc=False,
            )
            out[sid] = _raw_to_timeseries_df(df_raw, target_n_seconds=None)
        except Exception:
            out[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
    return out


def _set_session_timeouts(conn) -> None:
    """Best-effort session tuning to prevent long fetch disconnects."""
    try:
        cur = conn.cursor()
        cur.execute("SET SESSION wait_timeout=28800")
        cur.execute("SET SESSION interactive_timeout=28800")
        cur.execute(f"SET SESSION net_read_timeout={LONG_NET_TIMEOUT_SECONDS}")
        cur.execute(f"SET SESSION net_write_timeout={LONG_NET_TIMEOUT_SECONDS}")
        try:
            # 0 = no query timeout if supported (MySQL 5.7.8+)
            cur.execute("SET SESSION max_execution_time=0")
        except Exception:
            pass
        cur.close()
    except Exception:
        pass


def _fetch_rows_with_retries(
    *,
    query: str,
    params: list,
) -> list[dict]:
    """
    Execute a query with retries, streaming rows via fetchmany().

    This avoids pandas.read_sql() (which warns for mysql.connector) and follows
    the robust pattern used in Get_Data_09Dec25.py.
    """
    _ensure_mysql_available()
    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        conn = None
        cursor = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            _set_session_timeouts(conn)

            # Use a buffered cursor so we can safely fetch results without connection
            # state issues (and because our target payload is small: ~3400 rows).
            cursor = conn.cursor(dictionary=True, buffered=True)
            cursor.execute(query, params)

            rows: list[dict] = []
            fetched = 0
            while True:
                chunk = cursor.fetchmany(FETCH_CHUNK_SIZE)
                if not chunk:
                    break
                rows.extend(chunk)
                fetched += len(chunk)

            return rows
        except Exception as e:
            last_err = e
            # Backoff before retrying
            if attempt < MAX_RETRIES:
                delay = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
                time.sleep(delay)
        finally:
            try:
                if cursor is not None:
                    cursor.close()
            except Exception:
                pass
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

    # Exhausted retries
    if last_err is not None:
        raise last_err
    return []


def _query_db(
    *,
    start_time: datetime,
    end_time: datetime,
    limit_rows: int,
    sensor_id: Optional[str],
    order_desc: bool,
) -> pd.DataFrame:
    cols = ["id", "sensor_id", "timestamp", "b_x", "b_y", "b_z"]
    columns_str = ", ".join(cols)

    # Use ORDER BY id for performance (primary key), while still filtering by time.
    # This is much less likely to trigger long sort operations than ORDER BY timestamp.
    query = f"SELECT {columns_str} FROM {TABLE_NAME} WHERE timestamp >= %s AND timestamp <= %s"
    params: list = [start_time, end_time]
    if sensor_id:
        query += " AND sensor_id = %s"
        params.append(sensor_id)

    query += " ORDER BY id DESC" if order_desc else " ORDER BY id ASC"
    query += " LIMIT %s"
    params.append(int(limit_rows))

    rows = _fetch_rows_with_retries(query=query, params=params)
    return pd.DataFrame(rows)


def _raw_to_timeseries_df(df_raw: pd.DataFrame, target_n_seconds: Optional[int] = None) -> pd.DataFrame:
    """
    Convert raw DB rows into a 1 Hz time series:
    - bucket timestamps by second (floor to seconds)
    - average all samples within the same second
    - return columns [time_H, mag_H_nT]

    If target_n_seconds is provided, keep only the latest `target_n_seconds` rows.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["time_H", "mag_H_nT"])

    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=True).reset_index(drop=True)
    for c in ("b_x", "b_y", "b_z"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["mag_total_nT"] = (df["b_x"] ** 2 + df["b_y"] ** 2 + df["b_z"] ** 2) ** 0.5
    df = df.dropna(subset=["mag_total_nT"])

    # 1 Hz downsample by averaging within each second
    df["time_H"] = df["timestamp"].dt.floor("s")
    grouped = (
        df.groupby("time_H", as_index=False)["mag_total_nT"]
        .mean()
        .rename(columns={"mag_total_nT": "mag_H_nT"})
        .sort_values("time_H", ascending=True)
        .reset_index(drop=True)
    )

    if target_n_seconds is not None and target_n_seconds > 0 and len(grouped) > target_n_seconds:
        grouped = grouped.tail(int(target_n_seconds)).reset_index(drop=True)

    return grouped[["time_H", "mag_H_nT"]]


def get_timeseries_magnetic_data(
    session_id: Optional[str] = None,
    last_n_samples: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    hours: Optional[float] = None,
):
    """
    Return a DataFrame containing a scalar magnetic time series.

    Compatibility goal: keep the same signature and same column names used by `application.py`.

    Behavior requested:
    - Initial load should fetch the most recent 60 minutes and ~3400 rows.
    - Incremental loads (when `start_time` is provided) should fetch records after that time.

    Parameters are interpreted as:
    - If `end_time` is not provided: use current UTC time.
    - If `start_time` is not provided:
        - If `hours` is provided: start = end - hours
        - Else: default to last 60 minutes
    - Row limiting:
        - If `last_n_samples` is provided: cap to that.
        - Else: default to 3400 rows (per user requirement).
    """
    # Determine time bounds.
    #
    # Important: MySQL timestamps are commonly stored/handled as timezone-naive values.
    # To avoid driver issues with tz-aware datetimes, we use naive UTC for query params.
    if end_time is None:
        end_time = datetime.utcnow()
    else:
        end_time = pd.to_datetime(end_time).to_pydatetime()
        if getattr(end_time, "tzinfo", None) is not None:
            end_time = end_time.astimezone(timezone.utc).replace(tzinfo=None)

    if start_time is None:
        if hours is None:
            hours = 1.0
        start_time = end_time - timedelta(hours=float(hours))
    else:
        start_time = pd.to_datetime(start_time).to_pydatetime()
        if getattr(start_time, "tzinfo", None) is not None:
            start_time = start_time.astimezone(timezone.utc).replace(tzinfo=None)

    # Desired output size:
    # We return ONE averaged sample per second, so last_n_samples means "last N seconds".
    target_seconds: Optional[int]
    if last_n_samples is not None:
        target_seconds = int(last_n_samples)
    else:
        target_seconds = 3400

    # Decide whether to fetch most-recent rows (DESC) or forward-in-time rows (ASC).
    # - Initial "last 60 minutes" load: use DESC LIMIT N then sort ASC for plotting.
    # - Incremental fetch (start_time provided by caller): use ASC so we append correctly.
    is_incremental = start_time is not None and hours is None
    order_desc = not is_incremental

    # If incremental and caller did not specify last_n_samples, do not hard-cap by seconds.
    # (Caller usually wants "all new points since start_time".)
    if is_incremental and last_n_samples is None:
        target_seconds = None

    # Raw row limit: multiple samples per second, so fetch more than the target seconds.
    if target_seconds is None:
        limit_rows = 5000
    else:
        limit_rows = max(int(target_seconds) * 3, 12000)

    # Query
    _ensure_mysql_available()
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        sensor_id = _get_latest_sensor_id(conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    df_raw = _query_db(
        start_time=start_time,
        end_time=end_time,
        limit_rows=limit_rows,
        sensor_id=sensor_id,
        order_desc=order_desc,
    )

    # If we got no rows, it may be because the DB timestamps are in a different timezone
    # than the local machine clock. In that case, use the DB's latest timestamp as "now"
    # and fetch the last hour before that.
    if (df_raw is None or df_raw.empty) and hours is not None and start_time is not None and end_time is not None:
        try:
            db_max = _get_max_timestamp(sensor_id=sensor_id)
            if db_max is not None:
                end_time_db = db_max
                start_time_db = end_time_db - timedelta(hours=float(hours))
                df_raw = _query_db(
                    start_time=start_time_db,
                    end_time=end_time_db,
                    limit_rows=limit_rows,
                    sensor_id=sensor_id,
                    order_desc=order_desc,
                )
        except Exception:
            # Keep empty result; caller will handle.
            pass

    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["time_H", "mag_H_nT"])

    # Convert to 1 Hz series (average within each second)
    out = _raw_to_timeseries_df(df_raw, target_n_seconds=target_seconds)

    # Optional: persist fetched raw data for the session (debug/repro)
    if session_id:
        try:
            folder = os.path.join(APP_BASE, "sessions", session_id)
            os.makedirs(folder, exist_ok=True)
            out.to_csv(os.path.join(folder, "download_mag_db.csv"), index=False)
        except Exception:
            # Non-fatal: app should continue even if debug file can't be written.
            pass

    return out


