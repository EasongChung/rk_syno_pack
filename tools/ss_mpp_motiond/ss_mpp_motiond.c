#define _GNU_SOURCE

#include "ss_camera.h"
#include "ss_reclog.h"

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
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
#define FRAME_PAYLOAD_OFF 36
#define MIN_FRAME_PAYLOAD 1024
#define SAMPLE_STEP 32

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
	unsigned int threshold;
	unsigned int ratio_permille;
};

struct mpp_decoder {
	MppCtx ctx;
	MppApi *mpi;
};

struct options {
	const char *map_path;
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
};

static void usage(const char *prog)
{
	fprintf(stderr,
		"usage: %s (--camera-group N --map FILE | --camera-name NAME | --camera-dir DIR) [options]\n"
		"options:\n"
		"  --seconds N          run time, default 0 forever\n"
		"  --event-seconds N    event length, default 10\n"
		"  --cooldown N         write cooldown, default 30\n"
		"  --write-lag N        write at now-N, default 30\n"
		"  --threshold N        luma sample threshold, default 24\n"
		"  --ratio-permille N   changed sample ratio, default 30\n"
		"  --dry-run            detect and report without writing RecLog\n",
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

	fi->hdr.slot = get_le32(buf + 0);
	fi->hdr.aux_len = get_le32(buf + 4);
	fi->hdr.payload_len = get_le32(buf + 8);
	fi->hdr.serial = get_le32(buf + 12);
	fi->hdr.valid = get_le32(buf + 16);
	fi->hdr.frame_type = get_le32(buf + 20);
	fi->hdr.ts_lo = get_le32(buf + 24);
	fi->hdr.ts_hi = get_le32(buf + 28);
	fi->hdr.flags = get_le32(buf + 32);

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

static int cmp_frame_serial(const void *a, const void *b)
{
	const struct frame_info *fa = a;
	const struct frame_info *fb = b;

	if (fa->hdr.serial < fb->hdr.serial)
		return -1;
	if (fa->hdr.serial > fb->hdr.serial)
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
		if (parse_frame_segment(seg, &fi))
			break;

		if (!grp->frames || fi.hdr.serial < grp->min_serial)
			grp->min_serial = fi.hdr.serial;
		if (!grp->frames || fi.hdr.serial > grp->max_serial)
			grp->max_serial = fi.hdr.serial;
		grp->frames++;
	}

	grp->slots = grp->frames;
	grp->valid = grp->frames >= 2;
	return grp->frames;
}

static int find_groups(const struct shm_seg *segs, int n,
		       struct stream_group *groups, int max_groups)
{
	struct shm_seg sorted[MAX_SEGMENTS];
	int i, nr = 0;

	memcpy(sorted, segs, sizeof(*segs) * n);
	qsort(sorted, n, sizeof(sorted[0]), cmp_seg_key);

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

	qsort(frames, nr, sizeof(frames[0]), cmp_frame_serial);
	return nr;
}

static int copy_frame_payload(const struct frame_info *fi, uint8_t **out,
			      size_t *out_len)
{
	const uint8_t *buf;
	const void *addr;
	unsigned long long size;
	int ret;

	ret = attach_seg(fi->seg.shmid, &addr, &size);
	if (ret)
		return ret;

	if (fi->payload_off + fi->hdr.payload_len > size) {
		ret = -EINVAL;
		goto out_detach;
	}

	*out = malloc(fi->hdr.payload_len);
	if (!*out) {
		ret = -ENOMEM;
		goto out_detach;
	}

	buf = addr;
	memcpy(*out, buf + fi->payload_off, fi->hdr.payload_len);
	*out_len = fi->hdr.payload_len;
	ret = 0;

out_detach:
	shmdt(addr);
	return ret;
}

static int decoder_init(struct mpp_decoder *dec)
{
	RK_U32 split = 1;
	MPP_RET ret;

	memset(dec, 0, sizeof(*dec));

	ret = mpp_create(&dec->ctx, &dec->mpi);
	if (ret)
		return -EIO;

	dec->mpi->control(dec->ctx, MPP_DEC_SET_PARSER_SPLIT_MODE, &split);

	ret = mpp_init(dec->ctx, MPP_CTX_DEC, MPP_VIDEO_CodingHEVC);
	if (ret)
		return -EIO;

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

static void decoder_deinit(struct mpp_decoder *dec)
{
	if (dec->ctx) {
		MppPacket packet = NULL;
		RK_S32 timeout = 100;
		int i;

		if (!mpp_packet_init(&packet, NULL, 0)) {
			mpp_packet_set_eos(packet);
			for (i = 0; i < 100; i++) {
				if (dec->mpi->decode_put_packet(dec->ctx, packet) == MPP_OK)
					break;
				usleep(10000);
			}
			mpp_packet_deinit(&packet);
		}

		for (i = 0; i < 200; i++) {
			MppFrame frame = NULL;

			dec->mpi->control(dec->ctx, MPP_SET_OUTPUT_TIMEOUT, &timeout);
			if (dec->mpi->decode_get_frame(dec->ctx, &frame) || !frame) {
				usleep(10000);
				continue;
			}
			if (mpp_frame_get_info_change(frame))
				dec->mpi->control(dec->ctx,
						  MPP_DEC_SET_INFO_CHANGE_READY,
						  NULL);
			if (mpp_frame_get_eos(frame)) {
				mpp_frame_deinit(&frame);
				break;
			}
			mpp_frame_deinit(&frame);
		}

		mpp_destroy(dec->ctx);
	}
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
		printf("frame %ux%u stride=%ux%u samples=%u changed=%u ratio=%u motion=%u\n",
		       width, height, hstride, vstride, samples, changed, ratio,
		       *motion ? 1 : 0);
	} else {
		printf("frame %ux%u stride=%ux%u detector warmup\n",
		       width, height, hstride, vstride);
	}

	free(det->prev);
	det->prev = cur;
	det->prev_len = pos;
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

		dec->mpi->control(dec->ctx, MPP_SET_OUTPUT_TIMEOUT, &timeout);
		ret = dec->mpi->decode_get_frame(dec->ctx, &frame);
		if (ret || !frame)
			break;

		if (mpp_frame_get_info_change(frame))
			dec->mpi->control(dec->ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);
		else
			detector_frame(det, frame, motion);

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

static int mark_motion(const char *camera_dir, const struct options *opt,
		       time_t now)
{
	struct ss_reclog_result results[MAX_RESULTS];
	unsigned int nr = 0;
	time_t start, stop;
	int ret, i;

	if (now <= (time_t)opt->write_lag)
		return 0;

	start = now - opt->write_lag;
	stop = start + opt->event_seconds;

	ret = ss_reclog_mark_range_create_missing(camera_dir, start, stop, true,
						  opt->dry_run, results,
						  MAX_RESULTS, &nr);
	printf("motion event start=%ld stop=%ld dry_run=%u ret=%d\n",
	       (long)start, (long)stop, opt->dry_run ? 1 : 0, ret);
	for (i = 0; i < (int)nr; i++) {
		printf("  %s exists=%u touched=%u changed=%u err=%d\n",
		       results[i].path, results[i].exists, results[i].touched,
		       results[i].changed, results[i].err);
	}

	return ret;
}

static int resolve_camera_dir(const struct options *opt, char *camera_dir,
			      size_t len)
{
	struct ss_camera cams[MAX_CAMERAS];
	unsigned int nr = 0;
	int idx, ret;

	if (opt->camera_dir) {
		snprintf(camera_dir, len, "%s", opt->camera_dir);
		return 0;
	}

	ret = ss_camera_discover(cams, MAX_CAMERAS, &nr);
	if (ret)
		return ret;
	ss_camera_apply_group_map(cams, nr, opt->map_path);

	idx = ss_camera_find(cams, nr, -1, opt->camera_name, opt->camera_group);
	if (idx < 0)
		return -ENOENT;

	snprintf(camera_dir, len, "%s", cams[idx].camera_dir);
	return 0;
}

static int run_loop(const struct options *opt, const char *camera_dir)
{
	struct shm_seg segs[MAX_SEGMENTS];
	struct frame_info frames[MAX_SEGMENTS];
	struct mpp_decoder dec;
	struct detector det;
	unsigned long long first_key = 0;
	unsigned long long end = 0;
	unsigned int slots = 0;
	uint32_t last_serial = 0;
	time_t next_write = 0;
	bool started = false;
	int ret;

	memset(&det, 0, sizeof(det));
	det.threshold = opt->threshold;
	det.ratio_permille = opt->ratio_permille;

	ret = decoder_init(&dec);
	if (ret) {
		print_decoder_init_hint();
		return ret;
	}

	if (opt->seconds)
		end = (unsigned long long)time(NULL) + opt->seconds;

	while (!end || (unsigned long long)time(NULL) < end) {
		int n, nr, i;

		n = read_segments(segs, MAX_SEGMENTS);
		if (n < 0) {
			ret = n;
			break;
		}

		if (!first_key) {
			ret = select_group(segs, n, opt->camera_group,
					   &first_key, &slots);
			if (ret) {
				fprintf(stderr, "stream group %d not found\n",
					opt->camera_group);
				break;
			}
		}

		nr = collect_frames(segs, n, first_key, slots, frames,
				    MAX_SEGMENTS);
		for (i = 0; i < nr; i++) {
			struct frame_info *fi = &frames[i];
			uint8_t *payload = NULL;
			size_t payload_len = 0;
			bool motion = false;
			time_t now;

			if (fi->hdr.serial <= last_serial)
				continue;

			if (!started) {
				if (fi->hevc_type != 32 && fi->hdr.frame_type != 2) {
					last_serial = fi->hdr.serial;
					continue;
				}
				started = true;
			}

			if (copy_frame_payload(fi, &payload, &payload_len))
				continue;

			ret = decoder_put(&dec, payload, payload_len);
			free(payload);
			if (ret && ret != -EAGAIN)
				continue;

			decoder_drain(&dec, &det, &motion);
			last_serial = fi->hdr.serial;

			now = time(NULL);
			if (motion && now >= next_write) {
				mark_motion(camera_dir, opt, now);
				next_write = now + opt->cooldown;
			}
		}

		usleep(20000);
	}

	free(det.prev);
	decoder_deinit(&dec);
	return ret;
}

static int parse_args(int argc, char **argv, struct options *opt)
{
	int i;

	memset(opt, 0, sizeof(*opt));
	opt->camera_group = -1;
	opt->event_seconds = 10;
	opt->cooldown = 30;
	opt->write_lag = 30;
	opt->threshold = 24;
	opt->ratio_permille = 30;

	for (i = 1; i < argc; i++) {
		unsigned long long v;

		if (!strcmp(argv[i], "--map") && i + 1 < argc) {
			opt->map_path = argv[++i];
		} else if (!strcmp(argv[i], "--camera-group") && i + 1 < argc) {
			if (parse_u64(argv[++i], &v))
				return -EINVAL;
			opt->camera_group = (int)v;
		} else if (!strcmp(argv[i], "--camera-name") && i + 1 < argc) {
			opt->camera_name = argv[++i];
		} else if (!strcmp(argv[i], "--camera-dir") && i + 1 < argc) {
			opt->camera_dir = argv[++i];
		} else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) {
			if (parse_u64(argv[++i], &v))
				return -EINVAL;
			opt->seconds = (unsigned int)v;
		} else if (!strcmp(argv[i], "--event-seconds") && i + 1 < argc) {
			if (parse_u64(argv[++i], &v))
				return -EINVAL;
			opt->event_seconds = (unsigned int)v;
		} else if (!strcmp(argv[i], "--cooldown") && i + 1 < argc) {
			if (parse_u64(argv[++i], &v))
				return -EINVAL;
			opt->cooldown = (unsigned int)v;
		} else if (!strcmp(argv[i], "--write-lag") && i + 1 < argc) {
			if (parse_u64(argv[++i], &v))
				return -EINVAL;
			opt->write_lag = (unsigned int)v;
		} else if (!strcmp(argv[i], "--threshold") && i + 1 < argc) {
			if (parse_u64(argv[++i], &v))
				return -EINVAL;
			opt->threshold = (unsigned int)v;
		} else if (!strcmp(argv[i], "--ratio-permille") && i + 1 < argc) {
			if (parse_u64(argv[++i], &v))
				return -EINVAL;
			opt->ratio_permille = (unsigned int)v;
		} else if (!strcmp(argv[i], "--dry-run")) {
			opt->dry_run = true;
		} else {
			return -EINVAL;
		}
	}

	if (!opt->camera_dir && !opt->camera_name && opt->camera_group < 0)
		return -EINVAL;
	if (opt->camera_group >= 0 && !opt->map_path)
		return -EINVAL;
	if (!opt->event_seconds)
		opt->event_seconds = 1;

	return 0;
}

int main(int argc, char **argv)
{
	struct options opt;
	char camera_dir[SS_CAMERA_PATH_LEN];
	int ret;

	ret = parse_args(argc, argv, &opt);
	if (ret) {
		usage(argv[0]);
		return 2;
	}

	ret = resolve_camera_dir(&opt, camera_dir, sizeof(camera_dir));
	if (ret) {
		fprintf(stderr, "failed to resolve camera dir: %d\n", ret);
		return 1;
	}

	printf("camera_dir=%s group=%d dry_run=%u\n",
	       camera_dir, opt.camera_group, opt.dry_run ? 1 : 0);
	ret = run_loop(&opt, camera_dir);
	return ret ? 1 : 0;
}
