# Surveillance Station MPP Motion Daemon

Prototype daemon:

```text
Surveillance Station SysV shm
  -> RK MPP HEVC decoder
  -> luma frame-diff detector
  -> ss_reclog_mark_range()
```

Run it as `SurveillanceStation` or root so RecLog can be updated:

```sh
printf '0 鲁能门口\n1 西边\n' > /var/tmp/ss_motion_bridge.map

sudo -u SurveillanceStation ./ss_mpp_motiond \
  --camera-group 0 \
  --map /var/tmp/ss_motion_bridge.map \
  --seconds 60
```

The map is required for every selector. It lets `--camera-name` and
`--camera-dir` resolve to the corresponding SHM group as well as letting
`--camera-group` resolve to the camera recording directory.

The daemon ignores the current ring contents and starts at the next fresh
keyframe. Surveillance Station keeps a cached keyframe that can be much older
than the live P-frame tail; joining those frames produces an invalid reference
chain. `--bootstrap-cache` is available for diagnostics when startup latency is
more important than decode robustness.

The built-in detector is intentionally simple. It samples the decoded luma
plane and marks an event when enough samples changed compared with the previous
frame. A later RKNN/NPU detector can replace only the detector callback and
keep the same SHM/MPP/RecLog framework.

Runtime requirements:

- RecLog writes must run as `SurveillanceStation` or root.
- RK MPP needs `/dev/mpp_service` and a usable dma-buf allocator.
- On rk3566/rk3568, enable `CONFIG_DMABUF_HEAPS=y` and
  `CONFIG_DMABUF_HEAPS_SYSTEM=y`. Without `/dev/dma_heap/system`, the current
  MPP userspace falls back to DRM/CMA allocation; on DSM this can fail after
  boot because the default 32 MiB CMA pool is already exhausted.
- On DSM installations where `/dev/dma_heap/system` is mode `0600 root:root`,
  run the daemon as root or install a narrowly scoped device-permission rule.

Important parameters:

- `--write-lag N`: delay a detected event for N seconds before writing its
  original time range, default 60. If RecLog still has not reached that range,
  the event remains queued and is retried. The daemon never extends a partial
  live RecLog or invents recording state.
- `--state-file FILE`: override the pending-event state file. By default it is
  `/var/tmp/ss_mpp_motiond.<camera-path-hash>.pending`. Pending events are
  written with an atomic rename and restored after a daemon restart. The stored
  camera path is checked before any restored event can be written.
- `--event-seconds N`: event length to mark, default 10.
- `--cooldown N`: minimum seconds between writes, default 30.
- `--threshold N`: luma difference threshold per sample, default 24.
- `--ratio-permille N`: changed sample ratio, default 30 (3%).
- `--verbose`: print per-frame detector statistics. Production mode only logs
  stream selection and event writes.

The daemon retries the same packet when MPP reports a full input queue. It
does not advance the SHM serial until the packet is accepted. A five-second
decoded-frame watchdog resets MPP and rediscovers the selected SHM group after
a stream restart or stalled decoder. SHM slots are complete access units, so
MPP parser split mode must remain disabled.

Only a missing RecLog file or a live RecLog that has not reached the event
range is retried. Retries use a 5-to-60-second backoff and stop after 120
attempts. Permission, format, and I/O errors stop the daemon instead of filling
the queue indefinitely.
