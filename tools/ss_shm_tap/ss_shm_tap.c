#define _GNU_SOURCE

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

#define PROC_SYSVIPC_SHM "/proc/sysvipc/shm"
#define DEFAULT_MAX_SCAN (4U * 1024U * 1024U)
#define DEFAULT_MAX_DUMP (1024U * 1024U)
#define MAX_SEGMENTS 1024
#define MAX_GROUPS 64
#define FRAME_PAYLOAD_OFF 36
#define MIN_FRAME_PAYLOAD 1024
#define MAX_STREAM_SLOTS 64

struct shm_seg {
	unsigned long long key;
	int shmid;
	unsigned long long size;
	int nattch;
};

struct stats {
	unsigned long long nonzero;
	unsigned long long jpeg_soi;
	unsigned long long jpeg_eoi;
	unsigned long long h264_start3;
	unsigned long long h264_start4;
	unsigned long long h265_vps;
	unsigned long long ascii;
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
	bool has_key_frame;
	bool valid;
};

static void usage(const char *prog)
{
	fprintf(stderr,
		"usage:\n"
		"  %s --list [--prefix HEX]\n"
		"  %s --scan [--prefix HEX] [--max-scan BYTES]\n"
		"  %s --groups [--prefix HEX]\n"
		"  %s --frames [--prefix HEX]\n"
		"  %s --frames --group N | --base KEY\n"
		"  %s --capture FILE [--seconds N] [--prefix HEX]\n"
		"  %s --capture FILE [--seconds N] --group N | --base KEY\n"
		"  %s --dump-key KEY --out FILE [--max-dump BYTES]\n"
		"  %s --dump-shmid ID --out FILE [--max-dump BYTES]\n",
		prog, prog, prog, prog, prog, prog, prog, prog, prog);
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

static void print_hex(const uint8_t *buf, size_t len)
{
	size_t i;

	for (i = 0; i < len; i++) {
		if (i)
			putchar(' ');
		printf("%02x", buf[i]);
	}
}

static int read_segments(struct shm_seg *segs, int max)
{
	FILE *fp;
	char line[512];
	int n = 0;

	fp = fopen(PROC_SYSVIPC_SHM, "r");
	if (!fp) {
		perror(PROC_SYSVIPC_SHM);
		return -errno;
	}

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

static bool prefix_match(unsigned long long key, unsigned long long prefix)
{
	unsigned long long tmp = prefix;
	int nibbles = 0;

	if (!prefix)
		return true;

	while (tmp) {
		nibbles++;
		tmp >>= 4;
	}

	while (key >= (1ULL << (nibbles * 4)))
		key >>= 4;

	return key == prefix;
}

static void scan_buf(const uint8_t *buf, size_t len, struct stats *st)
{
	size_t i;

	memset(st, 0, sizeof(*st));

	for (i = 0; i < len; i++) {
		uint8_t c = buf[i];

		if (c)
			st->nonzero++;
		if (c >= 0x20 && c <= 0x7e)
			st->ascii++;

		if (i + 1 < len && buf[i] == 0xff && buf[i + 1] == 0xd8)
			st->jpeg_soi++;
		if (i + 1 < len && buf[i] == 0xff && buf[i + 1] == 0xd9)
			st->jpeg_eoi++;
		if (i + 2 < len && !buf[i] && !buf[i + 1] && buf[i + 2] == 1)
			st->h264_start3++;
		if (i + 4 < len && !buf[i] && !buf[i + 1] &&
		    !buf[i + 2] && buf[i + 3] == 1) {
			uint8_t nal = buf[i + 4];

			st->h264_start4++;
			if (((nal >> 1) & 0x3f) == 32)
				st->h265_vps++;
		}
	}
}

static int attach_seg(int shmid, const void **addr, unsigned long long *size)
{
	struct shmid_ds ds;
	void *ptr;

	if (shmctl(shmid, IPC_STAT, &ds) < 0) {
		perror("shmctl");
		return -errno;
	}

	ptr = shmat(shmid, NULL, SHM_RDONLY);
	if (ptr == (void *)-1) {
		perror("shmat");
		return -errno;
	}

	*addr = ptr;
	*size = ds.shm_segsz;
	return 0;
}

static int scan_segment(const struct shm_seg *seg, unsigned long long max_scan)
{
	const uint8_t *buf;
	const void *addr;
	unsigned long long size;
	struct stats st;
	size_t len;
	int ret;

	ret = attach_seg(seg->shmid, &addr, &size);
	if (ret)
		return ret;

	len = size < max_scan ? (size_t)size : (size_t)max_scan;
	buf = addr;
	scan_buf(buf, len, &st);

	printf("key=0x%08llx shmid=%d size=%llu nattch=%d scan=%zu ",
	       seg->key, seg->shmid, size, seg->nattch, len);
	printf("nonzero=%llu ascii=%llu jpeg=%llu/%llu start3=%llu start4=%llu h265_vps=%llu head=",
	       st.nonzero, st.ascii, st.jpeg_soi, st.jpeg_eoi,
	       st.h264_start3, st.h264_start4, st.h265_vps);
	print_hex(buf, len < 32 ? len : 32);
	putchar('\n');

	shmdt(addr);
	return 0;
}

static uint32_t get_le32(const uint8_t *p)
{
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
	       ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
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

static const char *hevc_name(uint8_t type)
{
	switch (type) {
	case 1:
		return "TRAIL_R";
	case 19:
		return "IDR_W_RADL";
	case 20:
		return "IDR_N_LP";
	case 21:
		return "CRA_NUT";
	case 32:
		return "VPS";
	case 33:
		return "SPS";
	case 34:
		return "PPS";
	case 39:
		return "PREFIX_SEI";
	case 40:
		return "SUFFIX_SEI";
	default:
		return "";
	}
}

static int print_frame_segment(const struct shm_seg *seg)
{
	struct frame_info fi;
	uint8_t h264_type;
	int ret;

	ret = parse_frame_segment(seg, &fi);
	if (ret)
		return 0;

	h264_type = fi.nal & 0x1f;

	printf("frame key=0x%08llx shmid=%d size=%llu slot=%u serial=%u "
	       "payload_len=%u payload_off=%zu valid=%u frame_type=%u "
	       "ts=%08x%08x nal=0x%02x hevc=%u:%s h264=%u\n",
	       fi.seg.key, fi.seg.shmid, fi.seg.size, fi.hdr.slot, fi.hdr.serial,
	       fi.hdr.payload_len, fi.payload_off, fi.hdr.valid,
	       fi.hdr.frame_type, fi.hdr.ts_hi, fi.hdr.ts_lo, fi.nal,
	       fi.hevc_type, hevc_name(fi.hevc_type),
	       h264_type);

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
		if (fi.hdr.frame_type == 2 || fi.hevc_type == 32)
			grp->has_key_frame = true;
		grp->frames++;
	}

	grp->slots = grp->frames;
	grp->valid = grp->frames >= 2;
	return grp->frames;
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

static int find_groups(const struct shm_seg *segs, int n,
		       unsigned long long prefix,
		       struct stream_group *groups, int max_groups)
{
	struct shm_seg sorted[MAX_SEGMENTS];
	int i, nr = 0;

	memcpy(sorted, segs, sizeof(*segs) * n);
	qsort(sorted, n, sizeof(sorted[0]), cmp_seg_key);

	for (i = 0; i < n && nr < max_groups; i++) {
		struct stream_group grp;
		unsigned long long first_key;

		if (!prefix_match(sorted[i].key, prefix))
			continue;

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

static int print_groups(const struct shm_seg *segs, int n,
			unsigned long long prefix)
{
	struct stream_group groups[MAX_GROUPS];
	int nr, i;

	nr = find_groups(segs, n, prefix, groups, MAX_GROUPS);
	for (i = 0; i < nr; i++) {
		struct stream_group *g = &groups[i];

		printf("group=%d ctrl=0x%08llx first_frame=0x%08llx "
		       "slots=%u frames=%u serial=%u..%u key_frame=%u\n",
		       i, g->ctrl_key, g->first_frame_key, g->slots,
		       g->frames, g->min_serial, g->max_serial,
		       g->has_key_frame ? 1 : 0);
	}

	return nr;
}

static int select_group(const struct shm_seg *segs, int n,
			unsigned long long prefix, int group_index,
			unsigned long long base_key,
			unsigned long long *first_key,
			unsigned int *slots)
{
	struct stream_group groups[MAX_GROUPS];
	int nr;

	*first_key = 0;
	*slots = 0;

	if (base_key) {
		const struct shm_seg *seg;
		struct frame_info fi;
		struct stream_group grp;
		unsigned long long key = 0;

		seg = find_seg_by_key(segs, n, base_key);
		if (seg && !parse_frame_segment(seg, &fi))
			key = base_key;
		else if (seg)
			key = ctrl_first_frame_key(seg);
		if (!key)
			key = base_key + 1;

		if (!group_frame_count(segs, n, key, &grp))
			return -ENOENT;
		*first_key = grp.first_frame_key;
		*slots = grp.slots;
		return 0;
	}

	if (group_index < 0)
		return 0;

	nr = find_groups(segs, n, prefix, groups, MAX_GROUPS);
	if (group_index >= nr)
		return -ENOENT;

	*first_key = groups[group_index].first_frame_key;
	*slots = groups[group_index].slots;
	return 0;
}

static bool key_selected(unsigned long long key, unsigned long long prefix,
			 unsigned long long first_key, unsigned int slots)
{
	if (first_key)
		return key >= first_key && key < first_key + slots;

	return prefix_match(key, prefix);
}

static int collect_frames(const struct shm_seg *segs, int n,
			  unsigned long long prefix,
			  unsigned long long first_key, unsigned int slots,
			  struct frame_info *frames, int max)
{
	int i, nr = 0;

	for (i = 0; i < n && nr < max; i++) {
		if (!key_selected(segs[i].key, prefix, first_key, slots))
			continue;
		if (!parse_frame_segment(&segs[i], &frames[nr]))
			nr++;
	}

	qsort(frames, nr, sizeof(frames[0]), cmp_frame_serial);
	return nr;
}

static int write_frame_payload(FILE *fp, const struct frame_info *fi)
{
	const uint8_t *buf;
	const void *addr;
	unsigned long long size;
	int ret;

	ret = attach_seg(fi->seg.shmid, &addr, &size);
	if (ret)
		return ret;

	buf = addr;
	if (fi->payload_off + fi->hdr.payload_len > size) {
		ret = -EINVAL;
		goto out;
	}

	if (fwrite(buf + fi->payload_off, 1, fi->hdr.payload_len, fp) !=
	    fi->hdr.payload_len) {
		perror("fwrite");
		ret = -EIO;
		goto out;
	}

	ret = 0;
out:
	shmdt(addr);
	return ret;
}

static unsigned long long now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (unsigned long long)ts.tv_sec * 1000ULL + ts.tv_nsec / 1000000ULL;
}

static int capture_stream(const char *path, unsigned int seconds,
			  unsigned long long prefix, int group_index,
			  unsigned long long base_key)
{
	struct shm_seg segs[MAX_SEGMENTS];
	struct frame_info frames[MAX_SEGMENTS];
	unsigned long long end;
	unsigned int written = 0;
	unsigned int skipped = 0;
	uint32_t last_serial = 0;
	bool started = false;
	bool selected = false;
	unsigned long long first_key = 0;
	unsigned int slots = 0;
	FILE *fp;
	int n;

	fp = fopen(path, "wb");
	if (!fp) {
		perror(path);
		return -errno;
	}

	end = now_ms() + (unsigned long long)seconds * 1000ULL;

	while (now_ms() < end) {
		int nr, i;

		n = read_segments(segs, MAX_SEGMENTS);
		if (n < 0) {
			fclose(fp);
			return n;
		}

		if (!selected) {
			if (select_group(segs, n, prefix, group_index, base_key,
					 &first_key, &slots)) {
				fclose(fp);
				fprintf(stderr, "stream group not found\n");
				return 1;
			}
			selected = true;
		}

		nr = collect_frames(segs, n, prefix, first_key, slots, frames,
				    MAX_SEGMENTS);
		for (i = 0; i < nr; i++) {
			struct frame_info *fi = &frames[i];

			if (fi->hdr.serial <= last_serial)
				continue;

			if (!started) {
				if (fi->hevc_type != 32 && fi->hdr.frame_type != 2) {
					last_serial = fi->hdr.serial;
					skipped++;
					continue;
				}
				started = true;
			}

			if (write_frame_payload(fp, fi))
				continue;
			last_serial = fi->hdr.serial;
			written++;
		}

		usleep(20000);
	}

	fclose(fp);
	printf("captured file=%s frames=%u skipped=%u started=%u\n",
	       path, written, skipped, started ? 1 : 0);
	return written ? 0 : 1;
}

static int dump_segment(int shmid, const char *out, unsigned long long max_dump)
{
	const void *addr;
	unsigned long long size;
	size_t len;
	FILE *fp;
	int ret;

	ret = attach_seg(shmid, &addr, &size);
	if (ret)
		return ret;

	len = size < max_dump ? (size_t)size : (size_t)max_dump;
	fp = fopen(out, "wb");
	if (!fp) {
		ret = -errno;
		perror(out);
		goto out_detach;
	}

	if (fwrite(addr, 1, len, fp) != len) {
		ret = -EIO;
		perror("fwrite");
	}

	fclose(fp);
	printf("dumped shmid=%d bytes=%zu out=%s\n", shmid, len, out);

out_detach:
	shmdt(addr);
	return ret;
}

static int find_shmid_by_key(unsigned long long key)
{
	int shmid;

	shmid = shmget((key_t)key, 0, 0);
	if (shmid < 0)
		perror("shmget");
	return shmid;
}

int main(int argc, char **argv)
{
	struct shm_seg segs[MAX_SEGMENTS];
	unsigned long long prefix = 0;
	unsigned long long max_scan = DEFAULT_MAX_SCAN;
	unsigned long long max_dump = DEFAULT_MAX_DUMP;
	unsigned long long dump_key = 0;
	unsigned long long base_key = 0;
	unsigned int seconds = 5;
	int dump_shmid = -1;
	int group_index = -1;
	const char *out = NULL;
	const char *capture = NULL;
	bool do_list = false;
	bool do_scan = false;
	bool do_frames = false;
	bool do_groups = false;
	unsigned long long selected_first_key = 0;
	unsigned int selected_slots = 0;
	int n, i;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--list")) {
			do_list = true;
		} else if (!strcmp(argv[i], "--scan")) {
			do_scan = true;
		} else if (!strcmp(argv[i], "--groups")) {
			do_groups = true;
		} else if (!strcmp(argv[i], "--frames")) {
			do_frames = true;
		} else if (!strcmp(argv[i], "--capture") && i + 1 < argc) {
			capture = argv[++i];
		} else if (!strcmp(argv[i], "--group") && i + 1 < argc) {
			unsigned long long tmp;

			if (parse_u64(argv[++i], &tmp))
				return 2;
			group_index = (int)tmp;
		} else if (!strcmp(argv[i], "--base") && i + 1 < argc) {
			if (parse_u64(argv[++i], &base_key))
				return 2;
		} else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) {
			unsigned long long tmp;

			if (parse_u64(argv[++i], &tmp))
				return 2;
			seconds = tmp ? (unsigned int)tmp : 1;
		} else if (!strcmp(argv[i], "--prefix") && i + 1 < argc) {
			if (parse_u64(argv[++i], &prefix))
				return 2;
		} else if (!strcmp(argv[i], "--max-scan") && i + 1 < argc) {
			if (parse_u64(argv[++i], &max_scan))
				return 2;
		} else if (!strcmp(argv[i], "--max-dump") && i + 1 < argc) {
			if (parse_u64(argv[++i], &max_dump))
				return 2;
		} else if (!strcmp(argv[i], "--dump-key") && i + 1 < argc) {
			if (parse_u64(argv[++i], &dump_key))
				return 2;
		} else if (!strcmp(argv[i], "--dump-shmid") && i + 1 < argc) {
			unsigned long long tmp;

			if (parse_u64(argv[++i], &tmp))
				return 2;
			dump_shmid = (int)tmp;
		} else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
			out = argv[++i];
		} else {
			usage(argv[0]);
			return 2;
		}
	}

	if (capture)
		return capture_stream(capture, seconds, prefix,
				      group_index, base_key) ? 1 : 0;

	if (dump_key || dump_shmid >= 0) {
		int shmid = dump_shmid;

		if (!out) {
			usage(argv[0]);
			return 2;
		}
		if (dump_key) {
			shmid = find_shmid_by_key(dump_key);
			if (shmid < 0)
				return 1;
		}
		return dump_segment(shmid, out, max_dump) ? 1 : 0;
	}

	n = read_segments(segs, MAX_SEGMENTS);
	if (n < 0)
		return 1;

	if (!do_list && !do_scan && !do_frames && !do_groups)
		do_scan = true;

	if (do_groups)
		print_groups(segs, n, prefix);

	if (do_frames && (group_index >= 0 || base_key)) {
		if (select_group(segs, n, prefix,
				 group_index, base_key, &selected_first_key,
				 &selected_slots)) {
			fprintf(stderr, "stream group not found\n");
			return 1;
		}
	}

	for (i = 0; i < n; i++) {
		if (!key_selected(segs[i].key, prefix, selected_first_key,
				  selected_slots))
			continue;
		if (do_list)
			printf("key=0x%08llx shmid=%d size=%llu nattch=%d\n",
			       segs[i].key, segs[i].shmid, segs[i].size,
			       segs[i].nattch);
		if (do_scan)
			scan_segment(&segs[i], max_scan);
		if (do_frames)
			print_frame_segment(&segs[i]);
	}

	return 0;
}
