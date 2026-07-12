#define _GNU_SOURCE

#include "ss_reclog.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <sys/sendfile.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define RECLOG_HEADER_SIZE 8
#define RECLOG_RECORD_SIZE 7
#define RECLOG_WINDOW_SECONDS (12 * 60 * 60)
#define RECLOG_FILE_SIZE (RECLOG_HEADER_SIZE + \
			  RECLOG_RECORD_SIZE * RECLOG_WINDOW_SECONDS)

static time_t reclog_base(time_t ts)
{
	return (ts / RECLOG_WINDOW_SECONDS) * RECLOG_WINDOW_SECONDS;
}

static int build_reclog_path(char *buf, size_t len, const char *camera_dir,
			     time_t ts)
{
	int ret;

	ret = snprintf(buf, len, "%s/@SSRECMETA/RecLog/%ld", camera_dir,
		       (long)reclog_base(ts));
	if (ret < 0 || (size_t)ret >= len)
		return -ENAMETOOLONG;

	return 0;
}

static int copy_file(const char *src, const char *dst)
{
	char buf[8192];
	int in, out;
	ssize_t n;
	int ret = 0;

	in = open(src, O_RDONLY);
	if (in < 0)
		return -errno;

	out = open(dst, O_WRONLY | O_CREAT | O_EXCL, 0644);
	if (out < 0) {
		ret = -errno;
		goto out_close_in;
	}

	while ((n = read(in, buf, sizeof(buf))) > 0) {
		char *p = buf;

		while (n > 0) {
			ssize_t w = write(out, p, n);

			if (w < 0) {
				ret = -errno;
				goto out_close_out;
			}
			p += w;
			n -= w;
		}
	}

	if (n < 0)
		ret = -errno;

out_close_out:
	close(out);
out_close_in:
	close(in);
	return ret;
}

static int backup_file(const char *path)
{
	char bak[640];
	time_t now = time(NULL);
	struct tm tm;
	int ret;

	localtime_r(&now, &tm);
	ret = snprintf(bak, sizeof(bak),
		       "%s.bak-%04d%02d%02d%02d%02d%02d",
		       path, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
		       tm.tm_hour, tm.tm_min, tm.tm_sec);
	if (ret < 0 || (size_t)ret >= sizeof(bak))
		return -ENAMETOOLONG;

	return copy_file(path, bak);
}

static int create_reclog_file(const char *path)
{
	unsigned char header[RECLOG_HEADER_SIZE] = { 0x07, 0, 0, 0, 0, 0, 0, 0 };
	char dir_buf[640];
	struct stat st;
	int fd, ret = 0;

	fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
	if (fd < 0)
		return -errno;

	if (write(fd, header, sizeof(header)) != (ssize_t)sizeof(header)) {
		ret = -EIO;
		goto out_unlink;
	}

	snprintf(dir_buf, sizeof(dir_buf), "%s", path);
	if (!stat(dirname(dir_buf), &st))
		fchown(fd, st.st_uid, st.st_gid);

	fsync(fd);
	close(fd);
	return 0;

out_unlink:
	close(fd);
	unlink(path);
	return ret;
}

static int ensure_reclog_size(int fd, time_t base, time_t stop,
			      bool create_missing)
{
	struct stat st;
	off_t need;

	if (!create_missing)
		return 0;
	if (stop <= base)
		return 0;
	if (stop > base + RECLOG_WINDOW_SECONDS)
		stop = base + RECLOG_WINDOW_SECONDS;

	need = RECLOG_HEADER_SIZE + (stop - base) * RECLOG_RECORD_SIZE;
	if (need > RECLOG_FILE_SIZE)
		need = RECLOG_FILE_SIZE;
	if (fstat(fd, &st) < 0)
		return -errno;
	if (st.st_size >= need)
		return 0;
	if (ftruncate(fd, need) < 0)
		return -errno;

	return 0;
}

static int mark_file(const char *path, time_t start, time_t stop, bool backup,
		     bool dry_run, bool create_missing,
		     struct ss_reclog_result *res)
{
	struct stat st;
	unsigned char *data;
	time_t base;
	unsigned int max_slots;
	unsigned int touched = 0;
	unsigned int changed = 0;
	bool created = false;
	int fd;
	int ret = 0;

	memset(res, 0, sizeof(*res));
	snprintf(res->path, sizeof(res->path), "%s", path);

	fd = open(path, dry_run ? O_RDONLY : O_RDWR);
	if (fd < 0) {
		if (errno == ENOENT && create_missing && !dry_run) {
			ret = create_reclog_file(path);
			if (!ret) {
				fd = open(path, O_RDWR);
				created = fd >= 0;
			}
		}
		if (fd >= 0) {
			ret = 0;
			goto opened;
		}
		res->exists = 0;
		res->err = ret ? ret : -errno;
		return 0;
	}

opened:
	res->exists = 1;

	if (fstat(fd, &st) < 0) {
		ret = -errno;
		goto out_close;
	}

	if (st.st_size < RECLOG_HEADER_SIZE) {
		ret = -EINVAL;
		goto out_close;
	}

	data = malloc(st.st_size);
	if (!data) {
		ret = -ENOMEM;
		goto out_close;
	}

	if (pread(fd, data, st.st_size, 0) != st.st_size) {
		ret = -EIO;
		goto out_free;
	}

	base = strtol(strrchr(path, '/') + 1, NULL, 10);
	if (!dry_run) {
		ret = ensure_reclog_size(fd, base, stop, create_missing);
		if (ret)
			goto out_free;
		if (fstat(fd, &st) < 0) {
			ret = -errno;
			goto out_free;
		}
		free(data);
		data = malloc(st.st_size);
		if (!data) {
			ret = -ENOMEM;
			goto out_close;
		}
		if (pread(fd, data, st.st_size, 0) != st.st_size) {
			ret = -EIO;
			goto out_free;
		}
	}
	max_slots = (st.st_size - RECLOG_HEADER_SIZE) / RECLOG_RECORD_SIZE;

	if (start < base)
		start = base;
	if (stop > base + (time_t)max_slots)
		stop = base + max_slots;

	while (start < stop) {
		off_t off = RECLOG_HEADER_SIZE +
			    (start - base) * RECLOG_RECORD_SIZE;

		if (off + 1 >= st.st_size)
			break;
		touched++;
		if (create_missing && !data[off])
			data[off] = 1;
		if (data[off] && data[off + 1] != 1) {
			data[off + 1] = 1;
			changed++;
		}
		start++;
	}

	if (changed && !dry_run) {
		if (backup && !created) {
			ret = backup_file(path);
			if (ret)
				goto out_free;
		}
		if (pwrite(fd, data, st.st_size, 0) != st.st_size) {
			ret = -EIO;
			goto out_free;
		}
		fsync(fd);
	}

	res->touched = touched;
	res->changed = changed;

out_free:
	free(data);
out_close:
	close(fd);
	res->err = ret;
	return ret;
}

static int mark_range(const char *camera_dir, time_t start, time_t stop,
		      bool backup, bool dry_run, bool create_missing,
		      struct ss_reclog_result *results,
		      unsigned int max_results, unsigned int *nr_results)
{
	unsigned int nr = 0;
	int ret = 0;

	if (!camera_dir || !results || !nr_results || start >= stop)
		return -EINVAL;

	while (start < stop) {
		struct ss_reclog_result *res;
		char path[512];
		time_t base = reclog_base(start);
		time_t part_stop = base + RECLOG_WINDOW_SECONDS;

		if (part_stop > stop)
			part_stop = stop;
		if (nr >= max_results)
			return -ENOSPC;

		ret = build_reclog_path(path, sizeof(path), camera_dir, start);
		if (ret)
			return ret;

		res = &results[nr++];
		ret = mark_file(path, start, part_stop, backup, dry_run,
				create_missing, res);
		if (ret)
			break;

		start = part_stop;
	}

	*nr_results = nr;
	return ret;
}

int ss_reclog_mark_range(const char *camera_dir, time_t start, time_t stop,
			 bool backup, bool dry_run,
			 struct ss_reclog_result *results,
			 unsigned int max_results, unsigned int *nr_results)
{
	return mark_range(camera_dir, start, stop, backup, dry_run, false,
			  results, max_results, nr_results);
}

int ss_reclog_mark_range_create_missing(const char *camera_dir,
					 time_t start, time_t stop,
					 bool backup, bool dry_run,
					 struct ss_reclog_result *results,
					 unsigned int max_results,
					 unsigned int *nr_results)
{
	return mark_range(camera_dir, start, stop, backup, dry_run, true,
			  results, max_results, nr_results);
}
