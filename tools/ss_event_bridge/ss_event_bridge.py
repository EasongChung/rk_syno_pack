#!/usr/bin/env python3
import argparse
import base64
import json
import os
import shutil
import sqlite3
import time


DEFAULT_RECORDING_DB = "/volume1/@surveillance/recording.db"
DEFAULT_DETECTION_DB = "/volume1/@surveillance/detection_event.db"
DEFAULT_RECORDING_SHARE = "/volume1/surveillance"
DEFAULT_MOTION_SECONDS = 12
MOTION_BUCKET_SECONDS = 1800
RECLOG_HEADER_SIZE = 8
RECLOG_RECORD_SIZE = 7
RECLOG_WINDOW_SECONDS = 12 * 60 * 60


def quote_ident(name):
    if not name.replace("_", "").isalnum():
        raise ValueError(f"invalid identifier: {name}")
    return '"' + name + '"'


def connect(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing database: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn, table):
    rows = conn.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    return [
        {
            "cid": r["cid"],
            "name": r["name"],
            "type": r["type"],
            "notnull": r["notnull"],
            "default": r["dflt_value"],
            "pk": r["pk"],
        }
        for r in rows
    ]


def inspect_db(args):
    with connect(args.db) as conn:
        tables = args.tables or [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        print(json.dumps({t: table_columns(conn, t) for t in tables},
                         ensure_ascii=False, indent=2))


def event_filter_sql(args):
    parts = ["trigger_label = 1", "stop_time > start_time"]
    params = []
    if args.cam_id is not None:
        parts.append("camera_id = ?")
        params.append(args.cam_id)
    if args.event_id is not None:
        parts.append("id = ?")
        params.append(args.event_id)
    return " AND ".join(parts), params


def list_events(args):
    where, params = event_filter_sql(args)
    sql = f"""
        SELECT id, camera_id, start_time, stop_time, trigger_label, recording,
               closing, path, video_width, video_height, framecount, filesize,
               video_type, audfmt, profile_types
        FROM event
        WHERE {where}
        ORDER BY id DESC
        LIMIT ?
    """
    params.append(args.limit)
    with connect(args.recording_db) as conn:
        rows = [dict(r) for r in conn.execute(sql, params)]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def fetch_event(conn, event_id):
    row = conn.execute(
        """
        SELECT id, ds_id, camera_id, start_time, stop_time, archived,
               video_width, video_height, framecount, path, filesize,
               video_type, audfmt, recording, closing, mark_as_del,
               profile_types
        FROM event
        WHERE id = ? AND trigger_label = 1 AND stop_time > start_time
        """,
        (event_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"no completed trigger_label event: {event_id}")
    return row


def fetch_covering_event(conn, camera_id, start_time, end_time):
    row = conn.execute(
        """
        SELECT id, ds_id, camera_id, start_time, stop_time, archived,
               video_width, video_height, framecount, path, filesize,
               video_type, audfmt, recording, closing, mark_as_del,
               profile_types
        FROM event
        WHERE camera_id = ?
          AND trigger_label = 1
          AND start_time <= ?
          AND stop_time >= ?
          AND stop_time > start_time
        ORDER BY start_time DESC
        LIMIT 1
        """,
        (camera_id, start_time, end_time),
    ).fetchone()
    return row


def day_period(ts):
    return time.strftime("%Y%m%d%p", time.localtime(ts))


def preview_dir(camera_dir, event):
    return os.path.join(
        camera_dir,
        "@SSRECMETA",
        "Preview",
        day_period(event["start_time"]),
        str(event["start_time"]),
    )


def thumbnail_path(camera_dir, event):
    return os.path.join(
        camera_dir,
        "@SSRECMETA",
        "Thumbnail",
        day_period(event["start_time"]),
        str(event["start_time"]),
    )


def list_preview_times(camera_dir, event):
    path = preview_dir(camera_dir, event)
    try:
        names = os.listdir(path)
    except OSError:
        return []

    values = []
    for name in names:
        try:
            ts = int(name)
        except ValueError:
            continue
        if event["start_time"] <= ts <= event["stop_time"]:
            values.append(ts)
    return sorted(values)


def choose_motion_window(event, camera_dir, motion_seconds):
    duration = max(1, event["stop_time"] - event["start_time"])
    window = max(1, min(motion_seconds, duration))

    if camera_dir:
        previews = list_preview_times(camera_dir, event)
        if previews:
            start = previews[0]
            end = min(event["stop_time"], start + window)
            if end <= start:
                end = min(event["stop_time"], start + 1)
            return start, end, previews

    start = event["start_time"]
    return start, min(event["stop_time"], start + window), []


def decode_thumb_data(data):
    data = data.strip()
    if data.startswith(b"/9j"):
        try:
            return base64.b64decode(data)
        except (ValueError, base64.binascii.Error):
            return None

    start_marker = data.find(b"\xff\xd8")
    if start_marker < 0:
        return None
    end_marker = data.find(b"\xff\xd9", start_marker + 2)
    if end_marker < 0:
        return None
    return data[start_marker:end_marker + 2]


def motion_thumb_file(recording_share, camera_id, start):
    bucket = (start // MOTION_BUCKET_SECONDS) * MOTION_BUCKET_SECONDS
    return os.path.join(
        recording_share,
        "@DetectionEvent",
        "Motion",
        f"{camera_id}-{bucket}",
    )


def reclog_base(ts):
    return (int(ts) // RECLOG_WINDOW_SECONDS) * RECLOG_WINDOW_SECONDS


def reclog_path(camera_dir, ts):
    return os.path.join(
        camera_dir,
        "@SSRECMETA",
        "RecLog",
        str(reclog_base(ts)),
    )


def patch_reclog_file(path, ranges, dry_run, backup):
    if not os.path.exists(path):
        return {"path": path, "exists": False, "changed": 0}

    try:
        base = int(os.path.basename(path).split("_", 1)[0])
    except ValueError:
        return {"path": path, "exists": True, "changed": 0, "error": "bad reclog name"}

    size = os.path.getsize(path)
    max_slots = max(0, (size - RECLOG_HEADER_SIZE) // RECLOG_RECORD_SIZE)
    potential = []
    touched = 0
    skipped = 0

    for start, stop in ranges:
        if int(start) < base or int(stop) > base + max_slots:
            return {
                "path": path,
                "exists": True,
                "changed": 0,
                "error": "range outside current RecLog",
            }

    flags = os.O_RDONLY if dry_run else os.O_RDWR
    fd = os.open(path, flags)
    try:
        for start, stop in ranges:
            for ts in range(int(start), int(stop)):
                off = RECLOG_HEADER_SIZE + (ts - base) * RECLOG_RECORD_SIZE
                record = os.pread(fd, 2, off)
                if len(record) != 2:
                    raise OSError("short RecLog read")
                touched += 1
                if record[0] == 0:
                    skipped += 1
                    continue
                if record[1] != 1:
                    potential.append(off + 1)

        if potential and not dry_run:
            if backup:
                stamp = time.strftime("%Y%m%d%H%M%S")
                bak = f"{path}.bak-{stamp}-{time.time_ns()}-{os.getpid()}"
                shutil.copy2(path, bak)
            for off in potential:
                record = os.pread(fd, 2, off - 1)
                if len(record) == 2 and record[0] and record[1] != 1:
                    os.pwrite(fd, b"\x01", off)
            os.fsync(fd)
    finally:
        os.close(fd)

    return {
        "path": path,
        "exists": True,
        "touched": touched,
        "skipped": skipped,
        "changed": len(potential),
    }


def sync_reclog_ranges(camera_dir, ranges, dry_run=False, backup=True):
    by_path = {}
    for start, stop in ranges:
        cur = int(start)
        stop = int(stop)
        while cur < stop:
            base = reclog_base(cur)
            part_stop = min(stop, base + RECLOG_WINDOW_SECONDS)
            by_path.setdefault(reclog_path(camera_dir, cur), []).append((cur, part_stop))
            cur = part_stop

    return [
        patch_reclog_file(path, ranges, dry_run, backup)
        for path, ranges in sorted(by_path.items())
    ]


def load_motion_ranges(conn, camera_id, start, stop, motion_ids=None, include_bad_thumb=False):
    where = [
        "camera_id = ?",
        "end_time > ?",
        "start_time < ?",
        "end_time > start_time",
    ]
    params = [camera_id, start, stop]
    if motion_ids:
        where.append(f"id IN ({','.join('?' for _ in motion_ids)})")
        params.extend(motion_ids)
    if not include_bad_thumb:
        where += [
            "thumb_token IS NOT NULL",
            "thumb_token != ''",
            "thumb_token NOT LIKE '%,0'",
        ]
    return [
        dict(row) for row in conn.execute(
            f"""
            SELECT id, camera_id, start_time, end_time, thumb_token
            FROM detection_event_motion
            WHERE {' AND '.join(where)}
            ORDER BY start_time
            """,
            params,
        )
    ]


def write_thumb_token(event, camera_dir, recording_share, start):
    if not camera_dir:
        return None

    thumb = thumbnail_path(camera_dir, event)
    if not os.path.exists(thumb):
        return None

    with open(thumb, "rb") as f:
        data = decode_thumb_data(f.read())
    if not data:
        return None

    path = motion_thumb_file(recording_share, event["camera_id"], start)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        offset = os.path.getsize(path)
    except OSError:
        offset = 0
    with open(path, "ab") as f:
        f.write(data)

    # ssapid/sswebstreamd may not run as root. Keep the generated container
    # readable; DSM-created files can later be tightened by the package itself.
    os.chmod(os.path.dirname(os.path.dirname(path)), 0o755)
    os.chmod(os.path.dirname(path), 0o755)
    os.chmod(path, 0o644)
    return f"{offset},{len(data)}"


def insert_alert(conn, event, camera_name, dry_run):
    exists = conn.execute(
        "SELECT 1 FROM alertevent WHERE event_id = ? LIMIT 1",
        (event["id"],),
    ).fetchone()
    if exists:
        return False

    params = {
        "ds_id": event["ds_id"],
        "camera_id": event["camera_id"],
        "camera_name": camera_name,
        "event_id": event["id"],
        "type": 1,
        "start_time": event["start_time"],
        "stop_time": event["stop_time"],
        "archived": event["archived"],
        "video_width": event["video_width"],
        "video_height": event["video_height"],
        "framecount": event["framecount"],
        "path": event["path"],
        "filesize": event["filesize"],
        "video_type": event["video_type"],
        "audfmt": event["audfmt"],
        "recording": event["recording"],
        "closing": event["closing"],
        "mark_as_del": event["mark_as_del"],
        "mark_as_file_del": 0,
        "viewed": 0,
        "device_type": 1,
        "sub_type": 0,
        "event_type": "motion_detection",
        "profile_types": event["profile_types"],
        "description": None,
    }
    cols = list(params)
    sql = (
        f"INSERT INTO alertevent ({', '.join(cols)}) "
        f"VALUES ({', '.join(':' + c for c in cols)})"
    )
    if not dry_run:
        conn.execute(sql, params)
    return True


def insert_detection(conn, event, camera_dir, recording_share, motion_seconds, dry_run):
    start, end, previews = choose_motion_window(event, camera_dir, motion_seconds)
    event_key = f"ss_event_bridge:event:{event['id']}"
    exists = conn.execute(
        """
        SELECT start_time, end_time, thumb_token
        FROM detection_event_motion
        WHERE description = ?
           OR (camera_id = ? AND start_time = ? AND end_time = ?)
        ORDER BY CASE WHEN description = ? THEN 0 ELSE 1 END
        LIMIT 1
        """,
        (event_key, event["camera_id"], start, end, event_key),
    ).fetchone()
    if exists:
        return {
            "inserted": False,
            "start": exists["start_time"],
            "end": exists["end_time"],
            "thumb_token": exists["thumb_token"],
        }

    quarter = start // 900
    thumb_token = None if dry_run else write_thumb_token(
        event, camera_dir, recording_share, start
    )
    if not dry_run:
        conn.execute(
            """
            INSERT INTO detection_event_motion
                (camera_id, ts_quarter, start_time, end_time, locked,
                 thumb_token, description)
            VALUES (?, ?, ?, ?, 0, NULL, ?)
            """,
            (event["camera_id"], quarter, start, end, event_key),
        )
        if thumb_token is not None:
            conn.execute(
                """
                UPDATE detection_event_motion
                SET thumb_token = ?
                WHERE id = last_insert_rowid()
                """,
                (thumb_token,),
            )
        conn.execute(
            """
            INSERT INTO detection_event_motion_count(camera_id, ts_quarter, total)
            VALUES (?, ?, 0)
            ON CONFLICT(camera_id, ts_quarter) DO NOTHING
            """,
                (event["camera_id"], quarter),
        )
        conn.execute(
            """
            UPDATE detection_event_motion_count
            SET total = (
                SELECT count(*) FROM detection_event_motion
                WHERE camera_id = ? AND ts_quarter = ?
            )
            WHERE camera_id = ? AND ts_quarter = ?
            """,
            (event["camera_id"], quarter, event["camera_id"], quarter),
        )
    return {"inserted": True, "start": start, "end": end, "thumb_token": thumb_token}


def bridge_event(args):
    if args.with_alert and not args.camera_name:
        raise SystemExit("--camera-name is required with --with-alert")

    with connect(args.recording_db) as rec:
        event = fetch_event(rec, args.event_id)
        alert_inserted = False
        if args.with_alert:
            with rec:
                alert_inserted = insert_alert(
                    rec, event, args.camera_name, args.dry_run
                )

    with connect(args.detection_db) as det:
        with det:
            motion = insert_detection(
                det, event, args.camera_dir, args.recording_share,
                args.motion_seconds, args.dry_run
            )
    reclog = []
    if args.sync_reclog and args.camera_dir:
        reclog = sync_reclog_ranges(
            args.camera_dir,
            [(motion["start"], motion["end"])],
            dry_run=args.dry_run,
            backup=args.reclog_backup,
        )

    print(json.dumps({
        "event_id": event["id"],
        "alert_inserted": alert_inserted,
        "motion": motion,
        "reclog": reclog,
        "camera_dir": args.camera_dir,
        "recording_share": args.recording_share,
        "dry_run": args.dry_run,
    }, ensure_ascii=False))


def bridge_pending(args):
    where, params = event_filter_sql(args)
    sql = f"""
        SELECT id FROM event e
        WHERE {where}
        ORDER BY id DESC
        LIMIT ?
    """
    params.append(args.limit)
    with connect(args.recording_db) as conn:
        ids = [r["id"] for r in conn.execute(sql, params)]

    for event_id in reversed(ids):
        sub = argparse.Namespace(**vars(args))
        sub.event_id = event_id
        bridge_event(sub)


def repair_thumbnails(args):
    with connect(args.detection_db) as det, connect(args.recording_db) as rec:
        rows = det.execute(
            """
            SELECT id, camera_id, start_time, end_time, thumb_token
            FROM detection_event_motion
            WHERE camera_id = ?
              AND (thumb_token IS NULL
                   OR thumb_token = ''
                   OR thumb_token LIKE '%,0')
            ORDER BY id DESC
            LIMIT ?
            """,
            (args.cam_id, args.limit),
        ).fetchall()

        repaired = []
        skipped = []
        with det:
            for row in rows:
                event = fetch_covering_event(
                    rec, row["camera_id"], row["start_time"], row["end_time"]
                )
                if event is None:
                    skipped.append({"id": row["id"], "reason": "no recording event"})
                    continue
                token = write_thumb_token(
                    event, args.camera_dir, args.recording_share, row["start_time"]
                )
                if token is None:
                    skipped.append({"id": row["id"], "reason": "no thumbnail"})
                    continue
                det.execute(
                    "UPDATE detection_event_motion SET thumb_token = ? WHERE id = ?",
                    (token, row["id"]),
                )
                repaired.append({"id": row["id"], "thumb_token": token})

    print(json.dumps({
        "repaired": repaired,
        "skipped": skipped,
    }, ensure_ascii=False))


def sync_reclog(args):
    with connect(args.detection_db) as det:
        rows = load_motion_ranges(
            det,
            args.cam_id,
            args.start,
            args.stop,
            motion_ids=args.motion_id,
            include_bad_thumb=args.include_bad_thumb,
        )

    ranges = [(row["start_time"], row["end_time"]) for row in rows]
    results = sync_reclog_ranges(
        args.camera_dir,
        ranges,
        dry_run=args.dry_run,
        backup=args.reclog_backup,
    )
    print(json.dumps({
        "rows": rows,
        "reclog": results,
        "dry_run": args.dry_run,
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Bridge Surveillance Station trigger_label events to motion indexes"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="print sqlite table schemas")
    p.add_argument("db")
    p.add_argument("tables", nargs="*")
    p.set_defaults(func=inspect_db)

    p = sub.add_parser("list", help="list completed trigger_label events")
    p.add_argument("--recording-db", default=DEFAULT_RECORDING_DB)
    p.add_argument("--cam-id", type=int)
    p.add_argument("--event-id", type=int)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=list_events)

    p = sub.add_parser("bridge", help="bridge one event.id")
    p.add_argument("--recording-db", default=DEFAULT_RECORDING_DB)
    p.add_argument("--detection-db", default=DEFAULT_DETECTION_DB)
    p.add_argument("--recording-share", default=DEFAULT_RECORDING_SHARE)
    p.add_argument("--camera-name")
    p.add_argument("--camera-dir")
    p.add_argument("--motion-seconds", type=int, default=DEFAULT_MOTION_SECONDS)
    p.add_argument("--event-id", type=int, required=True)
    p.add_argument(
        "--with-alert",
        action="store_true",
        help="also insert recording.db alertevent; normal motion timeline mainly uses detection_event_motion",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sync-reclog", action="store_true", default=True)
    p.add_argument("--no-sync-reclog", action="store_false", dest="sync_reclog")
    p.add_argument("--reclog-backup", action="store_true", default=True)
    p.add_argument("--no-reclog-backup", action="store_false", dest="reclog_backup")
    p.set_defaults(func=bridge_event)

    p = sub.add_parser("bridge-pending", help="bridge pending trigger_label events")
    p.add_argument("--recording-db", default=DEFAULT_RECORDING_DB)
    p.add_argument("--detection-db", default=DEFAULT_DETECTION_DB)
    p.add_argument("--recording-share", default=DEFAULT_RECORDING_SHARE)
    p.add_argument("--camera-name")
    p.add_argument("--camera-dir")
    p.add_argument("--motion-seconds", type=int, default=DEFAULT_MOTION_SECONDS)
    p.add_argument("--cam-id", type=int)
    p.add_argument("--event-id", type=int)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--with-alert", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sync-reclog", action="store_true", default=True)
    p.add_argument("--no-sync-reclog", action="store_false", dest="sync_reclog")
    p.add_argument("--reclog-backup", action="store_true", default=True)
    p.add_argument("--no-reclog-backup", action="store_false", dest="reclog_backup")
    p.set_defaults(func=bridge_pending)

    p = sub.add_parser("repair-thumbnails", help="repair empty motion thumb_token rows")
    p.add_argument("--recording-db", default=DEFAULT_RECORDING_DB)
    p.add_argument("--detection-db", default=DEFAULT_DETECTION_DB)
    p.add_argument("--recording-share", default=DEFAULT_RECORDING_SHARE)
    p.add_argument("--camera-dir", required=True)
    p.add_argument("--cam-id", type=int, required=True)
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=repair_thumbnails)

    p = sub.add_parser("sync-reclog", help="sync RecLog motion metadata from detection_event_motion")
    p.add_argument("--detection-db", default=DEFAULT_DETECTION_DB)
    p.add_argument("--camera-dir", required=True)
    p.add_argument("--cam-id", type=int, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--motion-id", action="append", type=int, default=[])
    p.add_argument("--include-bad-thumb", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reclog-backup", action="store_true", default=True)
    p.add_argument("--no-reclog-backup", action="store_false", dest="reclog_backup")
    p.set_defaults(func=sync_reclog)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
