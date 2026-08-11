#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import time

from ss_event_bridge import patch_reclog_file


DEFAULT_DETECTION_DB = "/volume1/@surveillance/detection_event.db"
DEFAULT_RECORDING_SHARE = "/volume1/surveillance"
RECLOG_HEADER_SIZE = 8
RECLOG_RECORD_SIZE = 7
RECLOG_WINDOW_SECONDS = 12 * 60 * 60


def connect(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing database: {path}")
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def day_period(ts):
    return time.strftime("%Y%m%d%p", time.localtime(ts))


def reclog_base(ts):
    return (int(ts) // RECLOG_WINDOW_SECONDS) * RECLOG_WINDOW_SECONDS


def camera_dir(recording_share, camera_name):
    return os.path.join(recording_share, camera_name)


def reclog_path(recording_share, camera_name, ts):
    base = reclog_base(ts)
    return os.path.join(
        camera_dir(recording_share, camera_name),
        "@SSRECMETA",
        "RecLog",
        str(base),
    )


def load_motion_rows(db, camera_id, start, stop, include_bad_thumb, motion_ids):
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
    with connect(db) as con:
        return [dict(row) for row in con.execute(
            f"""
            SELECT id, camera_id, start_time, end_time, thumb_token
            FROM detection_event_motion
            WHERE {' AND '.join(where)}
            ORDER BY start_time
            """,
            params,
        )]


def patch_file(path, ranges, dry_run, backup):
    return patch_reclog_file(path, ranges, dry_run, backup)


def patch_motion(args):
    rows = load_motion_rows(
        args.detection_db,
        args.camera_id,
        args.start,
        args.stop,
        args.include_bad_thumb,
        args.motion_id,
    )

    by_path = {}
    for row in rows:
        start = int(row["start_time"])
        stop = int(row["end_time"])
        cur = start
        while cur < stop:
            path = reclog_path(args.recording_share, args.camera_name, cur)
            base = reclog_base(cur)
            part_stop = min(stop, base + RECLOG_WINDOW_SECONDS)
            by_path.setdefault(path, []).append((cur, part_stop))
            cur = part_stop

    results = [
        patch_file(path, ranges, args.dry_run, args.backup)
        for path, ranges in sorted(by_path.items())
    ]

    print(json.dumps({
        "camera_id": args.camera_id,
        "camera_name": args.camera_name,
        "rows": rows,
        "results": results,
        "dry_run": args.dry_run,
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Surveillance Station RecLog motion metadata from detection_event_motion"
    )
    parser.add_argument("--detection-db", default=DEFAULT_DETECTION_DB)
    parser.add_argument("--recording-share", default=DEFAULT_RECORDING_SHARE)
    parser.add_argument("--camera-id", type=int, required=True)
    parser.add_argument("--camera-name", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--include-bad-thumb", action="store_true")
    parser.add_argument("--motion-id", action="append", type=int, default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup", action="store_true", default=True)
    parser.set_defaults(func=patch_motion)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
