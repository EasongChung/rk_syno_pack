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

The built-in detector is intentionally simple. It samples the decoded luma
plane and marks an event when enough samples changed compared with the previous
frame. A later RKNN/NPU detector can replace only the detector callback and
keep the same SHM/MPP/RecLog framework.

Runtime requirements:

- `ss_motion_bridge` RecLog writes must run as `SurveillanceStation` or root.
- RK MPP needs `/dev/mpp_service` and a usable dma-buf allocator.
- On rk3566/rk3568, enable `CONFIG_DMABUF_HEAPS=y` and
  `CONFIG_DMABUF_HEAPS_SYSTEM=y`. Without `/dev/dma_heap/system`, the current
  MPP userspace falls back to DRM/CMA allocation; on DSM this can fail after
  boot because the default 32 MiB CMA pool is already exhausted.

Important parameters:

- `--write-lag N`: write event at `now - N` seconds, default 30. This avoids
  writing the current second before Surveillance Station has flushed RecLog.
- `--event-seconds N`: event length to mark, default 10.
- `--cooldown N`: minimum seconds between writes, default 30.
- `--threshold N`: luma difference threshold per sample, default 24.
- `--ratio-permille N`: changed sample ratio, default 30 (3%).
