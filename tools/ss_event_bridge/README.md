# Surveillance Station Event Bridge

This is a diagnostic and repair tool, not a production event daemon. Database
schemas are version-specific; inspect and back up the target DSM databases
before using a command that is not `--dry-run`.

This helper is for the non-binary-patch diagnostic path: take a completed
recording row that already has `event.trigger_label = 1`, then write a matching
`detection_event_motion` row and thumbnail container entry.

It does not patch Surveillance Station binaries and does not bypass AME/HEVC
license checks.

Static analysis and WebAPI tests showed that normal motion timeline marks are
driven by two layers:

- `detection_event_motion`, `detection_event_motion_count`, and the
  `@DetectionEvent/Motion/<camera>-<bucket>` thumbnail container.
- per-camera `@SSRECMETA/RecLog/<base>` motion metadata. `RecordingPicker`
  reads this file to build `event_map`, so App/Web jump marks require RecLog
  metadata too.

`alertevent` is not the primary path for ordinary motion marks.

## Inspect schemas first

Run on DSM:

```sh
sudo python3 ss_event_bridge.py inspect \
  /volume1/@appstore/SurveillanceStation/recording.db \
  alertevent

sudo python3 ss_event_bridge.py inspect \
  /volume1/@appstore/SurveillanceStation/detection_event.db \
  detection_event_motion
```

Save the output before writing events. The insert path is schema-driven, but
the exact columns vary by Surveillance Station version.

## List candidate events

```sh
sudo python3 ss_event_bridge.py list --cam-id 1
```

## Bridge one event

```sh
sudo python3 ss_event_bridge.py bridge \
  --event-id 15 \
  --camera-name '鲁能门口' \
  --camera-dir '/volume1/surveillance/鲁能门口'
```

The command inserts matching rows into:

- `/volume1/@surveillance/detection_event.db`: `detection_event_motion` and
  `detection_event_motion_count`
- `/volume1/surveillance/<camera>/@SSRECMETA/RecLog`: per-second motion meta,
  when `--camera-dir` is set

It is idempotent for the same `event.id`.

If `--camera-dir` is set, the tool reads the recording thumbnail from
`@SSRECMETA/Thumbnail`, decodes base64 JPEG files when needed, appends the JPEG
to `@DetectionEvent/Motion`, and writes the resulting `thumb_token`.

To also insert a diagnostic `alertevent` row:

```sh
sudo python3 ss_event_bridge.py bridge --event-id 15 --camera-name '鲁能门口' --with-alert
```

## Bridge recent pending events

```sh
sudo python3 ss_event_bridge.py bridge-pending \
  --cam-id 1 \
  --camera-name '鲁能门口' \
  --camera-dir '/volume1/surveillance/鲁能门口'
```

Back up the databases before first use. This tool does not patch Surveillance
Station binaries and does not bypass AME/HEVC license checks.

This tool does not patch Surveillance Station binaries or change the camera
motion-detection source. It only backfills timeline index rows from events that
Surveillance Station has already marked with `trigger_label = 1`.

## Repair Empty Thumbnail Tokens

Older test rows or incomplete native rows may have `thumb_token` values like
`24,0`. Those rows can be listed by the API but `GetThumbnail` returns error
400. Repair a limited batch with:

```sh
sudo python3 ss_event_bridge.py repair-thumbnails \
  --cam-id 1 \
  --camera-dir '/volume1/surveillance/鲁能门口' \
  --limit 20
```

The repair command does not delete rows. It only fills missing or zero-length
thumbnail tokens when it can find a covering recording event and source
thumbnail for the explicitly selected camera.

## Sync RecLog From Existing Motion Rows

If `detection_event_motion` already has valid rows but Web/App timeline has no
jump marks, sync RecLog only:

```sh
sudo python3 ss_event_bridge.py sync-reclog \
  --camera-dir '/volume1/surveillance/鲁能门口' \
  --cam-id 1 \
  --start 1783440000 \
  --stop 1783526400 \
  --motion-id 28 \
  --motion-id 29 \
  --dry-run
```

Remove `--dry-run` to write. The command is idempotent and creates
`RecLog/<base>.bak-YYYYmmddHHMMSS` before modifying a RecLog file.
