# Surveillance Station Motion Bridge

C bridge for writing algorithm motion results into Surveillance Station
`@SSRECMETA/RecLog` metadata.

The detector should only report motion intervals. This bridge handles the
Synology RecLog layout:

- 12-hour RecLog file: `base = floor(ts / 43200) * 43200`
- header size: 8 bytes
- record size: 7 bytes per second
- byte 0: recording-present flag
- byte 1: motion/event flag

Example:

```sh
./ss_motion_bridge discover

./ss_motion_bridge mark \
  --camera-name '鲁能门口' \
  --start 1783429500 \
  --stop 1783429510

# Extend a partial RecLog, but only mark seconds whose recording bit is set.
./ss_motion_bridge mark \
  --camera-name '鲁能门口' \
  --start 1783429500 --stop 1783429510 \
  --create-missing

./ss_motion_bridge mark \
  --camera-group 0 \
  --map /var/tmp/ss_motion_bridge.map \
  --start 1783429500 \
  --stop 1783429510 \
  --dry-run
```

Path discovery is automatic:

- first scan `/volume*/*/*/@SSRECMETA/RecLog`, so the recording share does
  not need to be named `surveillance`;
- then, when running as root or `SurveillanceStation`, try to bind camera
  `id` from the discovered Surveillance Station `system.db`;
- fallback still works by `--camera-name` because the camera recording
  directory name is the Surveillance Station display/remark name.

Optional group map format:

```text
# shm_group camera_name
0 鲁能门口
1 西边
```

The detector can bind a shared-memory stream group to a camera name once, then
report motion by `--camera-group`.

Use `--dry-run` to verify the discovered RecLog path and changed second count
without modifying metadata. Real writes need permission to update the camera's
`@SSRECMETA/RecLog` files, so the bridge should run as root or as the
Surveillance Station package user in production.

Normal writes return `ERANGE` when the requested interval is outside a partial
RecLog. `--create-missing` may create or extend the file, but it never changes
the recording-present bit. `--force-recording` explicitly sets both recording
and motion bits and should only be used for controlled repair work.

Writes update only the affected per-second bytes. They do not rewrite or
truncate the complete RecLog while Surveillance Station is appending to it.

Framework direction:

```text
Surveillance Station shm group
  -> decoder / detector
  -> ss_reclog_mark_range(camera_dir, start, stop)
  -> original RecordingPicker builds event_map
```

This tool does not patch Surveillance Station binaries and does not write
SQLite rows. It only updates RecLog motion metadata for existing recordings.
