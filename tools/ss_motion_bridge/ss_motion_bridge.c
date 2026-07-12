#include "ss_reclog.h"
#include "ss_camera.h"

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_RESULTS 8
#define MAX_CAMERAS 64

static void usage(const char *prog)
{
	fprintf(stderr,
		"usage:\n"
		"  %s discover\n"
		"  %s mark (--camera-dir DIR | --camera-id ID | --camera-name NAME) "
		"--start UNIX --stop UNIX [--create-missing] [--no-backup] [--dry-run]\n"
		"  %s mark --camera-group N --map FILE --start UNIX --stop UNIX "
		"[--create-missing] [--no-backup] [--dry-run]\n",
		prog, prog, prog);
}

static int parse_time_arg(const char *s, time_t *out)
{
	char *end;
	long long v;

	errno = 0;
	v = strtoll(s, &end, 0);
	if (errno || end == s || *end || v < 0)
		return -EINVAL;
	*out = (time_t)v;
	return 0;
}

static int cmd_mark(int argc, char **argv)
{
	struct ss_reclog_result results[MAX_RESULTS];
	struct ss_camera cams[MAX_CAMERAS];
	const char *camera_dir = NULL;
	const char *camera_name = NULL;
	const char *map_path = NULL;
	time_t start = 0, stop = 0;
	unsigned int nr = 0;
	unsigned int nr_cams = 0;
	bool backup = true;
	bool dry_run = false;
	bool create_missing = false;
	int camera_id = -1;
	int camera_group = -1;
	int i, ret;

	for (i = 0; i < argc; i++) {
		if (!strcmp(argv[i], "--camera-dir") && i + 1 < argc) {
			camera_dir = argv[++i];
		} else if (!strcmp(argv[i], "--camera-name") && i + 1 < argc) {
			camera_name = argv[++i];
		} else if (!strcmp(argv[i], "--camera-id") && i + 1 < argc) {
			camera_id = atoi(argv[++i]);
		} else if (!strcmp(argv[i], "--camera-group") && i + 1 < argc) {
			camera_group = atoi(argv[++i]);
		} else if (!strcmp(argv[i], "--map") && i + 1 < argc) {
			map_path = argv[++i];
		} else if (!strcmp(argv[i], "--start") && i + 1 < argc) {
			if (parse_time_arg(argv[++i], &start))
				return -EINVAL;
		} else if (!strcmp(argv[i], "--stop") && i + 1 < argc) {
			if (parse_time_arg(argv[++i], &stop))
				return -EINVAL;
		} else if (!strcmp(argv[i], "--no-backup")) {
			backup = false;
		} else if (!strcmp(argv[i], "--create-missing")) {
			create_missing = true;
		} else if (!strcmp(argv[i], "--dry-run")) {
			dry_run = true;
		} else {
			return -EINVAL;
		}
	}

	if (!start || !stop || start >= stop)
		return -EINVAL;

	if (!camera_dir) {
		int idx;

		ret = ss_camera_discover(cams, MAX_CAMERAS, &nr_cams);
		if (ret)
			return ret;
		ss_camera_apply_group_map(cams, nr_cams, map_path);
		idx = ss_camera_find(cams, nr_cams, camera_id, camera_name,
				     camera_group);
		if (idx < 0)
			return -ENOENT;
		camera_dir = cams[idx].camera_dir;
	}

	if (create_missing)
		ret = ss_reclog_mark_range_create_missing(camera_dir, start, stop,
							  backup, dry_run,
							  results, MAX_RESULTS,
							  &nr);
	else
		ret = ss_reclog_mark_range(camera_dir, start, stop, backup, dry_run,
					   results, MAX_RESULTS, &nr);
	printf("{\"ok\":%s,\"dry_run\":%s,\"camera_dir\":\"%s\",\"results\":[",
	       ret ? "false" : "true", dry_run ? "true" : "false", camera_dir);
	for (i = 0; i < (int)nr; i++) {
		struct ss_reclog_result *r = &results[i];

		printf("%s{\"path\":\"%s\",\"exists\":%u,\"touched\":%u,"
		       "\"changed\":%u,\"err\":%d}",
		       i ? "," : "", r->path, r->exists, r->touched,
		       r->changed, r->err);
	}
	printf("],\"err\":%d}\n", ret);

	return ret ? 1 : 0;
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		usage(argv[0]);
		return 2;
	}

	if (!strcmp(argv[1], "discover")) {
		struct ss_camera cams[MAX_CAMERAS];
		unsigned int nr = 0;
		int ret;

		ret = ss_camera_discover(cams, MAX_CAMERAS, &nr);
		if (ret)
			return 1;
		ss_camera_print_json(cams, nr);
		return 0;
	}

	if (!strcmp(argv[1], "mark")) {
		int ret = cmd_mark(argc - 2, argv + 2);

		if (ret == -EINVAL) {
			usage(argv[0]);
			return 2;
		}
		return ret;
	}

	usage(argv[0]);
	return 2;
}
