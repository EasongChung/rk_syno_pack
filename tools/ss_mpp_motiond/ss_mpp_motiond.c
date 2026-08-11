#define _GNU_SOURCE

#include "ss_camera.h"
#include "ss_reclog.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <time.h>
#include <unistd.h>

#include <mpp_buffer.h>
#include <mpp_frame.h>
#include <mpp_packet.h>
#include <rk_mpi.h>

#define PROC_SYSVIPC_SHM "/proc/sysvipc/shm"
#define MAX_SEGMENTS 1024
#define MAX_GROUPS 64
#define MAX_CAMERAS 64
#define MAX_RESULTS 8
#define MAX_STREAM_SLOTS 64
#define MAX_PENDING_EVENTS 64
#define MAX_PENDING_RETRIES 120
#define PENDING_STATE_VERSION "SSMPP1"
#define FRAME_PAYLOAD_OFF 36
#define MIN_FRAME_PAYLOAD 1024
#define SAMPLE_STEP 32
#define DETECTOR_WARMUP_FRAMES 15

struct shm_seg {
	unsigned long long key;
	int shmid;
	unsigned long long size;
	int nattch;
};

struct frame_hdr {
	uint32_t slot;
	uint32_t aux_len;
	uint32_t payload_len;
	uint32_t serial;
	uint32_t valid;
	uint32_t frame_type;
	uint32_t ts_lo;
	uint32_t ts_hi;
	uint32_t flags;
};

struct frame_info {
	struct shm_seg seg;
	struct frame_hdr hdr;
	size_t payload_off;
	uint8_t nal;
	uint8_t hevc_type;
};

struct stream_group {
	unsigned long long ctrl_key;
	unsigned long long first_frame_key;
	unsigned int slots;
	unsigned int frames;
	uint32_t min_serial;
	uint32_t max_serial;
	bool valid;
};

struct detector {
	uint8_t *prev;
	size_t prev_len;
	uint64_t frames;
	unsigned int warmup_remaining;
	unsigned int threshold;
	unsigned int ratio_permille;
	bool verbose;
};

struct mpp_decoder {
	MppCtx ctx;
	MppApi *mpi;
	MppBufferGroup frame_group;
	bool submitted;
};

struct options {
	const char *map_path;
	const char *state_file;
	const char *camera_name;
	const char *camera_dir;
	int camera_group;
	unsigned int seconds;
	unsigned int event_seconds;
	unsigned int cooldown;
	unsigned int write_lag;
	unsigned int threshold;
	unsigned int ratio_permille;
	bool dry_run;
	bool verbose;
	bool bootstrap_cache;
};

struct pending_event {
	time_t start;
	time_t stop;
	time_t next_try;
	unsigned int retries;
	bool used;
};

static volatile sig_atomic_t stop_requested;

static void handle_signal(int signo)
{
	(void)signo;
	stop_requested = 1;
}

static void usage(const char *prog)
{
	fprintf(stderr,
		"usage: %s (--camera-group N | --camera-name NAME | --camera-dir DIR) --map FILE [options]\n"
		"options:\n"
		"  --seconds N          run time, default 0 forever\n"
		"  --event-seconds N    event length, default 10\n"
		"  --cooldown N         write cooldown, default 30\n"
		"  --write-lag N        defer RecLog write by N seconds, default 60\n"
		"  --threshold N        luma sample threshold, default 24\n"
		"  --ratio-permille N   changed sample ratio, default 30\n"
		"  --state-file FILE    pending-event state file\n"
		"  --dry-run            detect and report without writing RecLog\n"
		"  --verbose            print one line for every decoded frame\n"
		"  --bootstrap-cache    start from the cached keyframe (less robust)\n",
		prog);
}

static int parse_u64(const char *s, unsigned long long *val)
{
	char *end;

	errno = 0;
	*val = strtoull(s, &end, 0);
	if (errno || end == s || *end)
		return -EINVAL;
	return 0;
}

static int parse_uint(const char *s, unsigned int *val)
{
	unsigned long long parsed;
	int ret;

	ret = parse_u64(s, &parsed);
	if (ret)
		return ret;
	if (parsed > UINT_MAX)
		return -ERANGE;
	*val = (unsigned int)parsed;
	return 0;
}

static uint32_t camera_path_hash(const char *path)
{
	uint32_t hash = 2166136261U;

	while (*path) {
		hash ^= (unsigned char)*path++;
		hash *= 16777619U;
	}
	return hash;
}

static int read_segments(struct shm_seg *segs, int max)
{
	FILE *fp;
	char line[512];
	int n = 0;

	fp = fopen(PROC_SYSVIPC_SHM, "r");
	if (!fp)
		return -errno;

	if (!fgets(line, sizeof(line), fp)) {
		fclose(fp);
		return -EIO;
	}

	while (fgets(line, sizeof(line), fp)) {
		struct shm_seg seg;
		unsigned long long perms, cpid, lpid, uid, gid, cuid, cgid;
		unsigned long long atime, dtime, ctime, rss, swap;
		int ret;

		memset(&seg, 0, sizeof(seg));
		ret = sscanf(line,
			     "%lli %d %llo %llu %llu %llu %d %llu %llu %llu %llu %llu %llu %llu %llu %llu",
			     (long long *)&seg.key, &seg.shmid, &perms,
			     &seg.size, &cpid, &lpid, &seg.nattch, &uid, &gid,
			     &cuid, &cgid, &atime, &dtime, &ctime, &rss,
			     &swap);
		if (ret < 7)
			continue;
		if (n >= max)
			break;
		segs[n++] = seg;
	}

	fclose(fp);
	return n;
}

static uint32_t get_le32(const uint8_t *p)
{
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
	       ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void read_frame_header(const uint8_t *buf, struct frame_hdr *hdr)
{
	hdr->slot = get_le32(buf + 0);
	hdr->aux_len = get_le32(buf + 4);
	hdr->payload_len = get_le32(buf + 8);
	hdr->serial = get_le32(buf + 12);
	hdr->valid = get_le32(buf + 16);
	hdr->frame_type = get_le32(buf + 20);
	hdr->ts_lo = get_le32(buf + 24);
	hdr->ts_hi = get_le32(buf + 28);
	hdr->flags = get_le32(buf + 32);
}

static bool same_frame(const struct frame_hdr *a, const struct frame_hdr *b)
{
	return a->serial == b->serial &&
	       a->payload_len == b->payload_len &&
	       a->valid == 1 && b->valid == 1;
}

static uint64_t frame_timestamp(const struct frame_info *frame)
{
	return ((uint64_t)frame->hdr.ts_hi << 32) | frame->hdr.ts_lo;
}

static int attach_seg(int shmid, const void **addr, unsigned long long *size)
{
	struct shmid_ds ds;
	void *ptr;

	if (shmctl(shmid, IPC_STAT, &ds) < 0)
		return -errno;

	ptr = shmat(shmid, NULL, SHM_RDONLY);
	if (ptr == (void *)-1)
		return -errno;

	*addr = ptr;
	*size = ds.shm_segsz;
	return 0;
}

static int find_payload(const uint8_t *buf, size_t len, size_t *off)
{
	size_t i;

	for (i = FRAME_PAYLOAD_OFF; i + 5 < len && i < 128; i++) {
		if (!buf[i] && !buf[i + 1] && !buf[i + 2] &&
		    buf[i + 3] == 1) {
			*off = i;
			return 0;
		}
	}

	return -ENOENT;
}

static int parse_frame_segment(const struct shm_seg *seg, struct frame_info *fi)
{
	const uint8_t *buf;
	const void *addr;
	unsigned long long size;
	size_t off;
	int ret;

	if (seg->size < 64 || seg->size > 1024 * 1024)
		return -EINVAL;

	ret = attach_seg(seg->shmid, &addr, &size);
	if (ret)
		return ret;

	buf = addr;
	memset(fi, 0, sizeof(*fi));

	read_frame_header(buf, &fi->hdr);

	ret = find_payload(buf, (size_t)size, &off);
	if (ret)
		goto out;

	if (fi->hdr.valid != 1 || fi->hdr.payload_len < MIN_FRAME_PAYLOAD ||
	    fi->hdr.payload_len + off > size) {
		ret = -EINVAL;
		goto out;
	}

	fi->seg = *seg;
	fi->payload_off = off;
	fi->nal = buf[off + 4];
	fi->hevc_type = (fi->nal >> 1) & 0x3f;

out:
	shmdt(addr);
	return ret;
}

static int cmp_seg_key(const void *a, const void *b)
{
	const struct shm_seg *sa = a;
	const struct shm_seg *sb = b;

	if (sa->key < sb->key)
		return -1;
	if (sa->key > sb->key)
		return 1;
	return 0;
}

static int cmp_frame_time(const void *a, const void *b)
{
	const struct frame_info *fa = a;
	const struct frame_info *fb = b;
	uint64_t ta = frame_timestamp(fa);
	uint64_t tb = frame_timestamp(fb);

	if (ta < tb)
		return -1;
	if (ta > tb)
		return 1;
	return 0;
}

static const struct shm_seg *find_seg_by_key(const struct shm_seg *segs, int n,
					     unsigned long long key)
{
	int i;

	for (i = 0; i < n; i++) {
		if (segs[i].key == key)
			return &segs[i];
	}

	return NULL;
}

static unsigned long long ctrl_first_frame_key(const struct shm_seg *seg)
{
	const uint8_t *buf;
	const void *addr;
	unsigned long long size;
	unsigned long long key = 0;

	if (seg->size < 64 || seg->size > 65536)
		return 0;
	if (attach_seg(seg->shmid, &addr, &size))
		return 0;

	buf = addr;
	key = get_le32(buf + 0x34);
	shmdt(addr);
	return key;
}

static int group_frame_count(const struct shm_seg *segs, int n,
			     unsigned long long first_key,
			     struct stream_group *grp)
{
	unsigned int i;

	memset(grp, 0, sizeof(*grp));
	grp->first_frame_key = first_key;

	for (i = 0; i < MAX_STREAM_SLOTS; i++) {
		const struct shm_seg *seg;
		struct frame_info fi;

		seg = find_seg_by_key(segs, n, first_key + i);
		if (!seg)
			break;
		grp->slots = i + 1;
		if (parse_frame_segment(seg, &fi))
			continue;

		if (!grp->frames || fi.hdr.serial < grp->min_serial)
			grp->min_serial = fi.hdr.serial;
		if (!grp->frames || fi.hdr.serial > grp->max_serial)
			grp->max_serial = fi.hdr.serial;
		grp->frames++;
	}

	grp->valid = grp->frames >= 2;
	return (int)grp->frames;
}

static int find_groups(const struct shm_seg *segs, int n,
		       struct stream_group *groups, int max_groups)
{
	struct shm_seg sorted[MAX_SEGMENTS];
	int i, nr = 0;

	memcpy(sorted, segs, sizeof(*segs) * (size_t)n);
	qsort(sorted, (size_t)n, sizeof(sorted[0]), cmp_seg_key);

	for (i = 0; i < n && nr < max_groups; i++) {
		struct stream_group grp;
		unsigned long long first_key;

		first_key = ctrl_first_frame_key(&sorted[i]);
		if (!first_key)
			continue;
		if (!group_frame_count(sorted, n, first_key, &grp))
			continue;
		if (!grp.valid)
			continue;

		grp.ctrl_key = sorted[i].key;
		groups[nr++] = grp;
	}

	return nr;
}

static int select_group(const struct shm_seg *segs, int n, int group_index,
			unsigned long long *first_key, unsigned int *slots)
{
	struct stream_group groups[MAX_GROUPS];
	int nr;

	nr = find_groups(segs, n, groups, MAX_GROUPS);
	if (group_index < 0 || group_index >= nr)
		return -ENOENT;

	*first_key = groups[group_index].first_frame_key;
	*slots = groups[group_index].slots;
	printf("selected group=%d ctrl=0x%08llx first_frame=0x%08llx slots=%u\n",
	       group_index, groups[group_index].ctrl_key, *first_key, *slots);
	return 0;
}

static int collect_frames(const struct shm_seg *segs, int n,
			  unsigned long long first_key, unsigned int slots,
			  struct frame_info *frames, int max)
{
	int i, nr = 0;

	for (i = 0; i < n && nr < max; i++) {
		if (segs[i].key < first_key || segs[i].key >= first_key + slots)
			continue;
		if (!parse_frame_segment(&segs[i], &frames[nr]))
			nr++;
	}

	qsort(frames, (size_t)nr, sizeof(frames[0]), cmp_frame_time);
	return nr;
}

static int copy_frame_payload(const struct frame_info *fi, uint8_t **out,
			      size_t *out_len)
{
	const uint8_t *buf;
	const void *addr;
	unsigned long long size;
	struct frame_hdr before, after;
	size_t payload_off;
	int ret;

	ret = attach_seg(fi->seg.shmid, &addr, &size);
	if (ret)
		return ret;

	buf = addr;
	read_frame_header(buf, &before);
	if (!same_frame(&fi->hdr, &before) ||
	    find_payload(buf, (size_t)size, &payload_off) ||
	    payload_off + before.payload_len > size) {
		ret = -EINVAL;
		goto out_detach;
	}

	*out = malloc(before.payload_len);
	if (!*out) {
		ret = -ENOMEM;
		goto out_detach;
	}

	memcpy(*out, buf + payload_off, before.payload_len);
	__sync_synchronize();
	read_frame_header(buf, &after);
	if (!same_frame(&before, &after)) {
		free(*out);
		*out = NULL;
		ret = -EAGAIN;
		goto out_detach;
	}
	*out_len = before.payload_len;
	ret = 0;

out_detach:
	shmdt(addr);
	return ret;
}

static int decoder_init(struct mpp_decoder *dec, bool error_concealment)
{
	RK_U32 enable = 1;
	MPP_RET ret;

	memset(dec, 0, sizeof(*dec));

	ret = mpp_create(&dec->ctx, &dec->mpi);
	if (ret)
		return -EIO;

	ret = mpp_init(dec->ctx, MPP_CTX_DEC, MPP_VIDEO_CodingHEVC);
	if (ret) {
		mpp_destroy(dec->ctx);
		dec->ctx = NULL;
		return -EIO;
	}
	if (error_concealment) {
		dec->mpi->control(dec->ctx, MPP_DEC_SET_DISABLE_ERROR, &enable);
		dec->mpi->control(dec->ctx, MPP_DEC_SET_DISABLE_DPB_CHECK, &enable);
	}

	return 0;
}

static long read_meminfo_kb(const char *key)
{
	char line[256];
	FILE *fp;
	long value = -1;
	size_t key_len = strlen(key);

	fp = fopen("/proc/meminfo", "r");
	if (!fp)
		return -1;

	while (fgets(line, sizeof(line), fp)) {
		if (strncmp(line, key, key_len) || line[key_len] != ':')
			continue;
		value = strtol(line + key_len + 1, NULL, 10);
		break;
	}

	fclose(fp);
	return value;
}

static void print_decoder_init_hint(void)
{
	long cma_total = read_meminfo_kb("CmaTotal");
	long cma_free = read_meminfo_kb("CmaFree");

	if (cma_total >= 0 || cma_free >= 0) {
		fprintf(stderr, "mpp init failed; CmaTotal=%ld kB CmaFree=%ld kB\n",
			cma_total, cma_free);
		if (cma_free >= 0 && cma_free < 1024)
			fprintf(stderr,
				"mpp init hint: HEVC HAL needs DMA buffers; "
				"CMA is nearly exhausted. Enable DMA_HEAP system "
				"heap or increase CMA before running this daemon.\n");
	}
}

static int decoder_info_change(struct mpp_decoder *dec, MppFrame frame);

static void decoder_deinit(struct mpp_decoder *dec)
{
	MppPacket packet = NULL;
	RK_S32 timeout = 100;
	int i;

	if (!dec->ctx)
		return;

	if (dec->submitted && !mpp_packet_init(&packet, NULL, 0)) {
		mpp_packet_set_eos(packet);
		for (i = 0; i < 100; i++) {
			if (dec->mpi->decode_put_packet(dec->ctx, packet) == MPP_OK)
				break;
			usleep(1000);
		}
		mpp_packet_deinit(&packet);

		for (i = 0; i < 100; i++) {
			MppFrame frame = NULL;

			dec->mpi->control(dec->ctx, MPP_SET_OUTPUT_TIMEOUT, &timeout);
			if (dec->mpi->decode_get_frame(dec->ctx, &frame) || !frame)
				continue;
			if (mpp_frame_get_info_change(frame))
				decoder_info_change(dec, frame);
			if (mpp_frame_get_eos(frame)) {
				mpp_frame_deinit(&frame);
				break;
			}
			mpp_frame_deinit(&frame);
		}
	}

	dec->mpi->reset(dec->ctx);
	mpp_destroy(dec->ctx);
	dec->ctx = NULL;
	if (dec->frame_group) {
		mpp_buffer_group_put(dec->frame_group);
		dec->frame_group = NULL;
	}
}

static int decoder_info_change(struct mpp_decoder *dec, MppFrame frame)
{
	RK_U32 fast = 1;
	size_t frame_size = mpp_frame_get_buf_size(frame);
	RK_U32 size;
	MPP_RET ret;

	if (frame_size > UINT_MAX)
		return -EOVERFLOW;
	size = (RK_U32)frame_size;

	if (!dec->frame_group) {
		ret = mpp_buffer_group_get_internal(&dec->frame_group,
				MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_DMA32 |
				MPP_BUFFER_FLAGS_CACHABLE);
		if (ret)
			return -EIO;
	}

	ret = mpp_buffer_group_limit_config(dec->frame_group, size, 30);
	if (ret)
		return -EIO;
	ret = dec->mpi->control(dec->ctx, MPP_DEC_SET_EXT_BUF_GROUP,
				dec->frame_group);
	if (ret)
		return -EIO;
	ret = dec->mpi->control(dec->ctx, MPP_DEC_SET_PARSER_FAST_MODE, &fast);
	if (ret)
		return -EIO;
	ret = dec->mpi->control(dec->ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);
	return ret ? -EIO : 0;
}

static int detector_frame(struct detector *det, MppFrame frame, bool *motion)
{
	MppBuffer buf;
	uint8_t *ptr;
	uint8_t *cur;
	unsigned int width, height, hstride, vstride;
	unsigned int x, y, samples = 0, changed = 0;
	size_t need, pos = 0;

	*motion = false;

	if (mpp_frame_get_info_change(frame))
		return 0;
	if (mpp_frame_get_errinfo(frame) || mpp_frame_get_discard(frame))
		return 0;

	buf = mpp_frame_get_buffer(frame);
	if (!buf)
		return -EINVAL;

	ptr = mpp_buffer_get_ptr(buf);
	if (!ptr)
		return -EINVAL;

	width = mpp_frame_get_width(frame);
	height = mpp_frame_get_height(frame);
	hstride = mpp_frame_get_hor_stride(frame);
	vstride = mpp_frame_get_ver_stride(frame);
	if (!width || !height || hstride < width || vstride < height)
		return -EINVAL;

	need = ((width + SAMPLE_STEP - 1) / SAMPLE_STEP) *
	       ((height + SAMPLE_STEP - 1) / SAMPLE_STEP);
	cur = malloc(need);
	if (!cur)
		return -ENOMEM;

	for (y = 0; y < height; y += SAMPLE_STEP) {
		for (x = 0; x < width; x += SAMPLE_STEP) {
			uint8_t v = ptr[y * hstride + x];

			cur[pos] = v;
			if (det->prev && pos < det->prev_len) {
				int d = (int)v - det->prev[pos];

				if (d < 0)
					d = -d;
				if ((unsigned int)d >= det->threshold)
					changed++;
			}
			pos++;
			samples++;
		}
	}

	if (det->prev && samples) {
		unsigned int ratio = changed * 1000U / samples;

		if (ratio >= det->ratio_permille)
			*motion = true;
		if (det->verbose)
			printf("frame %ux%u stride=%ux%u samples=%u changed=%u ratio=%u motion=%u\n",
			       width, height, hstride, vstride, samples, changed,
			       ratio, *motion ? 1 : 0);
	} else if (det->verbose) {
		printf("frame %ux%u stride=%ux%u detector warmup\n",
		       width, height, hstride, vstride);
	}
	if (det->warmup_remaining) {
		det->warmup_remaining--;
		*motion = false;
	}

	free(det->prev);
	det->prev = cur;
	det->prev_len = pos;
	det->frames++;
	return 0;
}

static int decoder_drain(struct mpp_decoder *dec, struct detector *det,
			 bool *motion)
{
	int i;

	for (i = 0; i < 16; i++) {
		MppFrame frame = NULL;
		MPP_RET ret;
		RK_S32 timeout = 0;
		bool frame_motion = false;
		int detector_ret;

		dec->mpi->control(dec->ctx, MPP_SET_OUTPUT_TIMEOUT, &timeout);
		ret = dec->mpi->decode_get_frame(dec->ctx, &frame);
		if (ret || !frame)
			break;

		if (mpp_frame_get_info_change(frame)) {
			ret = decoder_info_change(dec, frame);
			mpp_frame_deinit(&frame);
			if (ret)
				return ret;
			continue;
		} else {
			detector_ret = detector_frame(det, frame, &frame_motion);
			if (detector_ret) {
				mpp_frame_deinit(&frame);
				return detector_ret;
			}
			if (frame_motion)
				*motion = true;
		}

		mpp_frame_deinit(&frame);
	}

	return 0;
}

static int decoder_put(struct mpp_decoder *dec, const uint8_t *data, size_t len)
{
	MppPacket packet = NULL;
	MPP_RET ret;

	ret = mpp_packet_init(&packet, (void *)data, len);
	if (ret)
		return -EIO;

	mpp_packet_set_pos(packet, (void *)data);
	mpp_packet_set_length(packet, len);

	ret = dec->mpi->decode_put_packet(dec->ctx, packet);
	mpp_packet_deinit(&packet);
	if (ret)
		return -EAGAIN;

	return 0;
}

static int decoder_submit(struct mpp_decoder *dec, struct detector *det,
			  const uint8_t *data, size_t len, bool *motion)
{
	int attempt;

	for (attempt = 0; attempt < 200 && !stop_requested; attempt++) {
		int ret = decoder_put(dec, data, len);

		if (!ret) {
			dec->submitted = true;
			return decoder_drain(dec, det, motion);
		}
		if (ret != -EAGAIN)
			return ret;

		ret = decoder_drain(dec, det, motion);
		if (ret)
			return ret;
		usleep(1000);
	}

	return -EAGAIN;
}

static unsigned int pending_count(const struct pending_event *events)
{
	unsigned int count = 0;
	int i;

	for (i = 0; i < MAX_PENDING_EVENTS; i++)
		count += events[i].used ? 1 : 0;
	return count;
}

static int save_pending(const struct pending_event *events, const char *path,
			const char *camera_dir)
{
	char tmp[PATH_MAX];
	FILE *fp = NULL;
	int fd = -1;
	int i, n, ret = 0;

	if (!pending_count(events)) {
		if (unlink(path) < 0 && errno != ENOENT)
			return -errno;
		return 0;
	}

	n = snprintf(tmp, sizeof(tmp), "%s.tmp.XXXXXX", path);
	if (n < 0 || (size_t)n >= sizeof(tmp))
		return -ENAMETOOLONG;

	fd = mkstemp(tmp);
	if (fd < 0)
		return -errno;
	fp = fdopen(fd, "w");
	if (!fp) {
		ret = -errno;
		close(fd);
		goto out_unlink;
	}

	if (fprintf(fp, "%s\ncamera %s\n", PENDING_STATE_VERSION,
		    camera_dir) < 0) {
		ret = errno ? -errno : -EIO;
		goto out_close;
	}
	for (i = 0; i < MAX_PENDING_EVENTS; i++) {
		if (!events[i].used)
			continue;
		if (fprintf(fp, "%" PRIdMAX " %" PRIdMAX " %u\n",
			    (intmax_t)events[i].start,
			    (intmax_t)events[i].stop,
			    events[i].retries) < 0) {
			ret = errno ? -errno : -EIO;
			goto out_close;
		}
	}
	if (fflush(fp) == EOF || fsync(fd) < 0) {
		ret = -errno;
		goto out_close;
	}
	if (fclose(fp) == EOF) {
		fp = NULL;
		ret = -errno;
		goto out_unlink;
	}
	fp = NULL;
	if (rename(tmp, path) < 0) {
		ret = -errno;
		goto out_unlink;
	}
	return 0;

out_close:
	fclose(fp);
out_unlink:
	unlink(tmp);
	return ret;
}

static int load_pending(struct pending_event *events, const char *path,
			const char *camera_dir, unsigned int *loaded_count)
{
	struct pending_event loaded[MAX_PENDING_EVENTS] = { 0 };
	char line[SS_CAMERA_PATH_LEN + 32];
	FILE *fp;
	int fd, ret = 0;
	unsigned int nr = 0;

	*loaded_count = 0;
	fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
	if (fd < 0)
		return errno == ENOENT ? 0 : -errno;
	fp = fdopen(fd, "r");
	if (!fp) {
		ret = -errno;
		close(fd);
		return ret;
	}

	if (!fgets(line, sizeof(line), fp) ||
	    strcmp(line, PENDING_STATE_VERSION "\n")) {
		ret = -EINVAL;
		goto out;
	}
	if (!fgets(line, sizeof(line), fp) || strncmp(line, "camera ", 7)) {
		ret = -EINVAL;
		goto out;
	}
	line[strcspn(line, "\r\n")] = '\0';
	if (strcmp(line + 7, camera_dir)) {
		ret = -EXDEV;
		goto out;
	}

	while (fgets(line, sizeof(line), fp)) {
		intmax_t start, stop;
		unsigned int retries;
		char extra;

		if (sscanf(line, "%" SCNdMAX " %" SCNdMAX " %u %c",
			   &start, &stop, &retries, &extra) != 3 ||
		    start < 0 || stop <= start ||
		    (intmax_t)(time_t)start != start ||
		    (intmax_t)(time_t)stop != stop ||
		    retries > MAX_PENDING_RETRIES) {
			ret = -EINVAL;
			goto out;
		}
		if (nr >= MAX_PENDING_EVENTS) {
			ret = -ENOSPC;
			goto out;
		}
		loaded[nr].start = (time_t)start;
		loaded[nr].stop = (time_t)stop;
		loaded[nr].retries = retries;
		loaded[nr].used = true;
		nr++;
	}
	if (ferror(fp)) {
		ret = -EIO;
		goto out;
	}

	memcpy(events, loaded, sizeof(loaded));
	*loaded_count = nr;
out:
	fclose(fp);
	return ret;
}

static int mark_motion(const char *camera_dir, const struct options *opt,
			       time_t start, time_t stop)
{
	struct ss_reclog_result results[MAX_RESULTS];
	unsigned int nr = 0;
	int ret, i;

	ret = ss_reclog_mark_range(camera_dir, start, stop, false,
				   opt->dry_run, results, MAX_RESULTS, &nr);
	printf("motion event start=%ld stop=%ld dry_run=%u ret=%d\n",
	       (long)start, (long)stop, opt->dry_run ? 1 : 0, ret);
	for (i = 0; i < (int)nr; i++) {
		printf("  %s exists=%u touched=%u recorded=%u skipped=%u changed=%u err=%d\n",
		       results[i].path, results[i].exists, results[i].touched,
		       results[i].recorded, results[i].skipped,
		       results[i].changed, results[i].err);
	}

	return ret;
}

static int queue_motion(struct pending_event *events, time_t now,
			unsigned int duration)
{
	int i;

	for (i = 0; i < MAX_PENDING_EVENTS; i++) {
		if (events[i].used)
			continue;
		events[i].start = now > (time_t)duration ? now - duration : now;
		events[i].stop = now;
		events[i].next_try = 0;
		events[i].retries = 0;
		events[i].used = true;
		return 0;
	}

	return -ENOSPC;
}

static bool retryable_mark_error(int ret)
{
	return ret == -ENOENT || ret == -ERANGE;
}

static unsigned int retry_delay(unsigned int retries)
{
	unsigned int shift = retries > 5 ? 4 : retries - 1;
	unsigned int delay = 5U << shift;

	return delay > 60 ? 60 : delay;
}

static int flush_pending(struct pending_event *events, const char *camera_dir,
			 const char *state_path, const struct options *opt,
			 time_t now)
{
	bool dirty = false;
	int fatal = 0;
	int i;

	for (i = 0; i < MAX_PENDING_EVENTS; i++) {
		int ret;

		if (!events[i].used ||
		    now < events[i].stop + (time_t)opt->write_lag ||
		    now < events[i].next_try)
			continue;

		ret = mark_motion(camera_dir, opt, events[i].start, events[i].stop);
		if (!ret) {
			events[i].used = false;
			dirty = true;
			continue;
		}
		if (!retryable_mark_error(ret)) {
			fatal = ret;
			break;
		}
		if (events[i].retries >= MAX_PENDING_RETRIES) {
			fprintf(stderr,
				"motion event retry limit reached start=%ld stop=%ld\n",
				(long)events[i].start, (long)events[i].stop);
			fatal = -ETIMEDOUT;
			break;
		}
		events[i].retries++;
		events[i].next_try = now + retry_delay(events[i].retries);
		dirty = true;
	}

	if (dirty && !opt->dry_run) {
		int ret = save_pending(events, state_path, camera_dir);

		if (ret)
			return ret;
	}
	return fatal;
}

static int resolve_camera(const struct options *opt, char *camera_dir,
			  size_t len, int *stream_group)
{
	struct ss_camera cams[MAX_CAMERAS];
	unsigned int nr = 0;
	int idx, ret;

	ret = ss_camera_discover(cams, MAX_CAMERAS, &nr);
	if (ret)
		return ret;
	ret = ss_camera_apply_group_map(cams, nr, opt->map_path);
	if (ret)
		return ret;

	if (opt->camera_dir)
		idx = ss_camera_find_path(cams, nr, opt->camera_dir);
	else
		idx = ss_camera_find(cams, nr, -1, opt->camera_name,
				     opt->camera_group);
	if (idx < 0)
		return -ENOENT;
	if (cams[idx].group < 0)
		return -ENXIO;

	snprintf(camera_dir, len, "%s", cams[idx].camera_dir);
	*stream_group = cams[idx].group;
	return 0;
}

static int run_loop(const struct options *opt, const char *camera_dir,
		    int stream_group, const char *state_path)
{
	struct shm_seg segs[MAX_SEGMENTS];
	struct frame_info frames[MAX_SEGMENTS];
	struct mpp_decoder dec;
	struct detector det;
	struct pending_event pending[MAX_PENDING_EVENTS] = { 0 };
	unsigned long long first_key = 0;
	unsigned long long end = 0;
	unsigned int slots = 0;
	uint64_t last_timestamp = 0;
	unsigned int idle_loops = 0;
	uint64_t last_decoded_frames = 0;
	time_t next_write = 0;
	bool have_timestamp = false;
	bool baseline_set = false;
	bool started = false;
	int ret = 0;
	int i;

	memset(&det, 0, sizeof(det));
	det.threshold = opt->threshold;
	det.ratio_permille = opt->ratio_permille;
	det.verbose = opt->verbose;
	det.warmup_remaining = DETECTOR_WARMUP_FRAMES;

	if (!opt->dry_run) {
		unsigned int loaded = 0;

		ret = load_pending(pending, state_path, camera_dir, &loaded);
		if (ret) {
			fprintf(stderr, "failed to load pending state %s: %d\n",
				state_path, ret);
			return ret;
		}
		if (loaded)
			printf("loaded %u pending motion events from %s\n",
			       loaded, state_path);
		for (i = 0; i < MAX_PENDING_EVENTS; i++) {
			time_t event_next_write;

			if (!pending[i].used)
				continue;
			event_next_write = pending[i].stop +
				(time_t)opt->cooldown;
			if (event_next_write > next_write)
				next_write = event_next_write;
		}
	}

	ret = decoder_init(&dec, opt->bootstrap_cache);
	if (ret) {
		print_decoder_init_hint();
		return ret;
	}

	if (opt->seconds)
		end = (unsigned long long)time(NULL) + opt->seconds;

	while (!stop_requested &&
	       (!end || (unsigned long long)time(NULL) < end)) {
		int n, nr;
		bool discard_bootstrap_tail = false;

		n = read_segments(segs, MAX_SEGMENTS);
		if (n < 0) {
			ret = n;
			break;
		}

		if (!first_key) {
			ret = select_group(segs, n, stream_group,
					   &first_key, &slots);
			if (ret) {
				fprintf(stderr, "stream group %d not found\n",
					stream_group);
				break;
			}
		}
		if (!find_seg_by_key(segs, n, first_key)) {
			first_key = 0;
			slots = 0;
			last_timestamp = 0;
			have_timestamp = false;
			baseline_set = false;
			started = false;
			free(det.prev);
			det.prev = NULL;
			det.prev_len = 0;
			det.warmup_remaining = DETECTOR_WARMUP_FRAMES;
			dec.mpi->reset(dec.ctx);
			dec.submitted = false;
			usleep(20000);
			continue;
		}

		nr = collect_frames(segs, n, first_key, slots, frames,
				    MAX_SEGMENTS);
		if (!baseline_set && nr > 0) {
			baseline_set = true;
			if (opt->bootstrap_cache) {
				discard_bootstrap_tail = true;
				if (opt->verbose)
					printf("bootstrapping from cached keyframe\n");
			} else {
				last_timestamp = frame_timestamp(&frames[nr - 1]);
				have_timestamp = true;
			}
			if (opt->verbose && !opt->bootstrap_cache)
				printf("waiting for next keyframe after timestamp=%" PRIu64 "\n",
				       last_timestamp);
			if (!opt->bootstrap_cache) {
				usleep(20000);
				continue;
			}
		}
		for (i = 0; i < nr; i++) {
			struct frame_info *fi = &frames[i];
			uint8_t *payload = NULL;
			size_t payload_len = 0;
			bool motion = false;
			bool keyframe = fi->hevc_type == 32 || fi->hdr.frame_type == 2;
			uint64_t timestamp = frame_timestamp(fi);
			time_t now;
			int submit_ret;

			if (have_timestamp && timestamp <= last_timestamp)
				continue;

			if (!started) {
				if (!keyframe) {
					last_timestamp = timestamp;
					have_timestamp = true;
					continue;
				}
				started = true;
			} else if (discard_bootstrap_tail && !keyframe) {
				last_timestamp = timestamp;
				have_timestamp = true;
				continue;
			}
			if (copy_frame_payload(fi, &payload, &payload_len))
				continue;

			submit_ret = decoder_submit(&dec, &det, payload, payload_len,
						    &motion);
			free(payload);
			if (submit_ret == -EAGAIN)
				break;
			if (submit_ret) {
				fprintf(stderr, "decoder submit failed: %d\n", submit_ret);
				ret = submit_ret;
				stop_requested = 1;
				break;
			}
			last_timestamp = timestamp;
			have_timestamp = true;

			now = time(NULL);
			if (motion && now >= next_write) {
				ret = queue_motion(pending, now, opt->event_seconds);
				if (!ret) {
					if (!opt->dry_run)
						ret = save_pending(pending, state_path,
								   camera_dir);
					if (ret) {
						fprintf(stderr,
							"failed to save pending state %s: %d\n",
							state_path, ret);
						stop_requested = 1;
						break;
					}
					printf("queued motion start=%ld stop=%ld\n",
					       (long)(now - opt->event_seconds), (long)now);
					next_write = now + opt->cooldown;
				} else {
					fprintf(stderr, "motion queue full\n");
					stop_requested = 1;
					break;
				}
			}
		}
		if (ret)
			break;
		ret = flush_pending(pending, camera_dir, state_path, opt,
				    time(NULL));
		if (ret) {
			fprintf(stderr, "failed to flush pending motion events: %d\n",
				ret);
			break;
		}

		if (!started) {
			idle_loops = 0;
		} else if (det.frames != last_decoded_frames) {
			last_decoded_frames = det.frames;
			idle_loops = 0;
		} else if (++idle_loops >= 250) {
			fprintf(stderr, "stream stalled; rediscovering group %d\n",
				stream_group);
			first_key = 0;
			slots = 0;
			last_timestamp = 0;
			have_timestamp = false;
			baseline_set = false;
			started = false;
			idle_loops = 0;
			free(det.prev);
			det.prev = NULL;
			det.prev_len = 0;
			det.warmup_remaining = DETECTOR_WARMUP_FRAMES;
			dec.mpi->reset(dec.ctx);
			dec.submitted = false;
		}

		usleep(20000);
	}

	free(det.prev);
	decoder_deinit(&dec);
	return ret;
}

static int parse_args(int argc, char **argv, struct options *opt)
{
	int i, selectors = 0;

	memset(opt, 0, sizeof(*opt));
	opt->camera_group = -1;
	opt->event_seconds = 10;
	opt->cooldown = 30;
	opt->write_lag = 60;
	opt->threshold = 24;
	opt->ratio_permille = 30;
	opt->bootstrap_cache = false;

	for (i = 1; i < argc; i++) {
		unsigned long long v;

		if (!strcmp(argv[i], "--map") && i + 1 < argc) {
			opt->map_path = argv[++i];
		} else if (!strcmp(argv[i], "--state-file") && i + 1 < argc) {
			opt->state_file = argv[++i];
		} else if (!strcmp(argv[i], "--camera-group") && i + 1 < argc) {
			if (parse_u64(argv[++i], &v))
				return -EINVAL;
			if (v > INT_MAX)
				return -ERANGE;
			opt->camera_group = (int)v;
			selectors++;
		} else if (!strcmp(argv[i], "--camera-name") && i + 1 < argc) {
			opt->camera_name = argv[++i];
			selectors++;
		} else if (!strcmp(argv[i], "--camera-dir") && i + 1 < argc) {
			opt->camera_dir = argv[++i];
			selectors++;
		} else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) {
			if (parse_uint(argv[++i], &opt->seconds))
				return -EINVAL;
		} else if (!strcmp(argv[i], "--event-seconds") && i + 1 < argc) {
			if (parse_uint(argv[++i], &opt->event_seconds))
				return -EINVAL;
		} else if (!strcmp(argv[i], "--cooldown") && i + 1 < argc) {
			if (parse_uint(argv[++i], &opt->cooldown))
				return -EINVAL;
		} else if (!strcmp(argv[i], "--write-lag") && i + 1 < argc) {
			if (parse_uint(argv[++i], &opt->write_lag))
				return -EINVAL;
		} else if (!strcmp(argv[i], "--threshold") && i + 1 < argc) {
			if (parse_uint(argv[++i], &opt->threshold))
				return -EINVAL;
		} else if (!strcmp(argv[i], "--ratio-permille") && i + 1 < argc) {
			if (parse_uint(argv[++i], &opt->ratio_permille))
				return -EINVAL;
		} else if (!strcmp(argv[i], "--dry-run")) {
			opt->dry_run = true;
		} else if (!strcmp(argv[i], "--verbose")) {
			opt->verbose = true;
		} else if (!strcmp(argv[i], "--bootstrap-cache")) {
			opt->bootstrap_cache = true;
		} else if (!strcmp(argv[i], "--wait-keyframe")) {
			opt->bootstrap_cache = false;
		} else {
			return -EINVAL;
		}
	}

	if (selectors != 1 || !opt->map_path ||
	    (opt->state_file && !opt->state_file[0]))
		return -EINVAL;
	if (opt->ratio_permille > 1000 || opt->threshold > 255)
		return -EINVAL;
	if (!opt->event_seconds)
		opt->event_seconds = 1;

	return 0;
}

int main(int argc, char **argv)
{
	struct options opt;
	char camera_dir[SS_CAMERA_PATH_LEN];
	char default_state_path[PATH_MAX];
	const char *state_path;
	int stream_group = -1;
	int n, ret;

	setvbuf(stdout, NULL, _IOLBF, 0);
	ret = parse_args(argc, argv, &opt);
	if (ret) {
		usage(argv[0]);
		return 2;
	}

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	ret = resolve_camera(&opt, camera_dir, sizeof(camera_dir),
			     &stream_group);
	if (ret) {
		fprintf(stderr, "failed to resolve camera dir: %d\n", ret);
		return 1;
	}
	if (opt.state_file) {
		state_path = opt.state_file;
	} else {
		n = snprintf(default_state_path, sizeof(default_state_path),
			     "/var/tmp/ss_mpp_motiond.%08" PRIx32 ".pending",
			     camera_path_hash(camera_dir));
		if (n < 0 || (size_t)n >= sizeof(default_state_path)) {
			fprintf(stderr, "pending state path is too long\n");
			return 1;
		}
		state_path = default_state_path;
	}

	printf("camera_dir=%s group=%d state=%s dry_run=%u\n",
	       camera_dir, stream_group, state_path, opt.dry_run ? 1 : 0);
	ret = run_loop(&opt, camera_dir, stream_group, state_path);
	return ret ? 1 : 0;
}
