# Surveillance Station SHM Tap

Read-only SysV shared-memory scanner for Surveillance Station stream buffers.

It never writes to shared memory. The first pass is intentionally simple:

`--capture` requires exactly one stream selector: `--group`, `--base`, or a
non-zero `--prefix`. This prevents multiple camera streams from being mixed
into one Annex-B output file.

```sh
./ss_shm_tap --list
./ss_shm_tap --groups
./ss_shm_tap --frames --group 0
./ss_shm_tap --capture /tmp/live.hevc --seconds 5 --group 0
./ss_shm_tap --scan --prefix 0x1818 --max-scan 4194304
```

The scanner reports JPEG markers, H.264/H.265 Annex-B start-code markers, and
basic byte statistics so we can identify which segments carry live frames.

`--capture` polls the frame slots and writes an Annex-B HEVC stream in timestamp
order. It waits for a key/config frame by default, so the output is suitable for
feeding into FFmpeg, MPP, or a later motion detector without opening the camera
RTSP stream again.

For multiple cameras, use `--groups` first. Each group is one Surveillance
Station stream FIFO. `--group N` binds `--frames` and `--capture` to one camera
stream. The group list is discovered from shared-memory control blocks at
runtime; the key prefix is not hard-coded.

Example with two cameras:

```text
group=0 ctrl=0x181824d0 first_frame=0x181824d1 slots=15 ...
group=1 ctrl=0x18182669 first_frame=0x1818266a slots=10 ...
```

Use `--group 0` for the first stream and `--group 1` for the second. For
detection code, always bind to a group. A broad `--prefix` scan is only for
diagnostics and can mix multiple cameras. `--base KEY` is still available for
manual debugging when you already know a control-block key.
