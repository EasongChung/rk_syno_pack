#define _GNU_SOURCE

#include "ss_camera.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_VOLUMES 32
#define MAX_DB_PATHS 16

static int path_exists(const char *path)
{
	struct stat st;

	return !stat(path, &st);
}

static int is_dir(const char *path)
{
	struct stat st;

	return !stat(path, &st) && S_ISDIR(st.st_mode);
}

static int path_join(char *buf, size_t len, const char *a, const char *b)
{
	int ret;

	ret = snprintf(buf, len, "%s/%s", a, b);
	if (ret < 0 || (size_t)ret >= len)
		return -ENAMETOOLONG;

	return 0;
}

static void json_escape(const char *s)
{
	putchar('"');
	for (; *s; s++) {
		if (*s == '"' || *s == '\\')
			printf("\\%c", *s);
		else if ((unsigned char)*s < 0x20)
			printf("\\u%04x", (unsigned char)*s);
		else
			putchar(*s);
	}
	putchar('"');
}

static int add_camera(struct ss_camera *cams, unsigned int max_cams,
		      unsigned int *nr_cams, const char *name,
		      const char *camera_dir)
{
	struct ss_camera *cam;
	unsigned int i;

	for (i = 0; i < *nr_cams; i++) {
		if (!strcmp(cams[i].camera_dir, camera_dir))
			return 0;
	}

	if (*nr_cams >= max_cams)
		return -ENOSPC;

	cam = &cams[(*nr_cams)++];
	memset(cam, 0, sizeof(*cam));
	cam->id = -1;
	cam->group = -1;
	snprintf(cam->name, sizeof(cam->name), "%s", name);
	snprintf(cam->camera_dir, sizeof(cam->camera_dir), "%s", camera_dir);
	return 0;
}

static int scan_recording_share(const char *share, struct ss_camera *cams,
				unsigned int max_cams, unsigned int *nr_cams)
{
	DIR *dir;
	struct dirent *de;

	dir = opendir(share);
	if (!dir)
		return -errno;

	while ((de = readdir(dir))) {
		char camera_dir[SS_CAMERA_PATH_LEN];
		char reclog[SS_CAMERA_PATH_LEN];

		if (de->d_name[0] == '.')
			continue;
		if (de->d_name[0] == '@')
			continue;

		if (path_join(camera_dir, sizeof(camera_dir), share, de->d_name))
			continue;
		if (path_join(reclog, sizeof(reclog), camera_dir,
			      "@SSRECMETA/RecLog"))
			continue;
		if (!is_dir(reclog))
			continue;

		add_camera(cams, max_cams, nr_cams, de->d_name, camera_dir);
	}

	closedir(dir);
	return 0;
}

static void scan_volume_shares(const char *volume, struct ss_camera *cams,
			       unsigned int max_cams, unsigned int *nr_cams)
{
	DIR *dir;
	struct dirent *de;

	dir = opendir(volume);
	if (!dir)
		return;

	while ((de = readdir(dir))) {
		char share[SS_CAMERA_PATH_LEN];

		if (de->d_name[0] == '.')
			continue;
		if (de->d_name[0] == '@')
			continue;
		if (path_join(share, sizeof(share), volume, de->d_name))
			continue;
		if (is_dir(share))
			scan_recording_share(share, cams, max_cams, nr_cams);
	}

	closedir(dir);
}

static int scan_volume_recordings(struct ss_camera *cams, unsigned int max_cams,
				  unsigned int *nr_cams)
{
	DIR *root;
	struct dirent *de;

	root = opendir("/");
	if (!root)
		return -errno;

	while ((de = readdir(root))) {
		char volume[SS_CAMERA_PATH_LEN];

		if (strncmp(de->d_name, "volume", 6))
			continue;
		snprintf(volume, sizeof(volume), "/%s", de->d_name);
		if (is_dir(volume))
			scan_volume_shares(volume, cams, max_cams, nr_cams);
	}

	closedir(root);
	return 0;
}

static char *trim(char *s)
{
	char *e;

	while (isspace((unsigned char)*s))
		s++;
	e = s + strlen(s);
	while (e > s && isspace((unsigned char)e[-1]))
		*--e = 0;
	return s;
}

static void bind_name_to_id(struct ss_camera *cams, unsigned int nr_cams,
			    const char *name, int id)
{
	unsigned int i;

	if (!name || !*name || id < 0)
		return;

	for (i = 0; i < nr_cams; i++) {
		if (!strcmp(cams[i].name, name)) {
			cams[i].id = id;
			return;
		}
	}
}

static int add_db_path(char paths[][SS_CAMERA_PATH_LEN], unsigned int *nr_paths,
		       const char *path)
{
	unsigned int i;

	if (!path_exists(path))
		return 0;

	for (i = 0; i < *nr_paths; i++) {
		if (!strcmp(paths[i], path))
			return 0;
	}

	if (*nr_paths >= MAX_DB_PATHS)
		return -ENOSPC;

	snprintf(paths[(*nr_paths)++], SS_CAMERA_PATH_LEN, "%s", path);
	return 0;
}

static void find_system_dbs(char paths[][SS_CAMERA_PATH_LEN],
			    unsigned int *nr_paths)
{
	DIR *root;
	struct dirent *de;

	*nr_paths = 0;
	add_db_path(paths, nr_paths,
		    "/var/packages/SurveillanceStation/target/system.db");

	root = opendir("/");
	if (!root)
		return;

	while ((de = readdir(root))) {
		char path[SS_CAMERA_PATH_LEN];

		if (strncmp(de->d_name, "volume", 6))
			continue;
		snprintf(path, sizeof(path),
			 "/%s/@appstore/SurveillanceStation/system.db",
			 de->d_name);
		add_db_path(paths, nr_paths, path);
	}

	closedir(root);
}

static int try_bind_ids_from_query(struct ss_camera *cams, unsigned int nr_cams,
				   const char *db, const char *table,
				   const char *id_col, const char *name_col)
{
	char cmd[1024];
	char line[512];
	FILE *fp;
	unsigned int bound = 0;

	/*
	 * Surveillance Station keeps these databases mode 0600. This succeeds
	 * when the bridge runs as root/SurveillanceStation and silently falls
	 * back to name/path-only discovery otherwise.
	 */
	snprintf(cmd, sizeof(cmd),
		 "sqlite3 -readonly -separator '|' '%s' "
		 "\"select %s,%s from %s where %s is not null and %s is not null;\" 2>/dev/null",
		 db, id_col, name_col, table, id_col, name_col);

	fp = popen(cmd, "r");
	if (!fp)
		return 0;

	while (fgets(line, sizeof(line), fp)) {
		char *sep, *name;
		int id;

		sep = strchr(line, '|');
		if (!sep)
			continue;
		*sep++ = 0;
		id = atoi(trim(line));
		name = trim(sep);
		bind_name_to_id(cams, nr_cams, name, id);
		bound++;
	}

	pclose(fp);
	return bound;
}

static void try_bind_ids_from_db(struct ss_camera *cams, unsigned int nr_cams,
				 const char *db)
{
	static const char *tables[] = {
		"camera",
		"cameras",
		"device",
		"devices",
		"cam",
		"ip_camera",
		"recording_camera",
	};
	static const char *id_cols[] = {
		"id",
		"camera_id",
		"cam_id",
		"ds_id",
	};
	static const char *name_cols[] = {
		"name",
		"camera_name",
		"cam_name",
		"display_name",
		"new_name",
		"nickname",
		"remark",
		"remarks",
	};
	unsigned int t, i, n;

	for (t = 0; t < sizeof(tables) / sizeof(tables[0]); t++) {
		for (i = 0; i < sizeof(id_cols) / sizeof(id_cols[0]); i++) {
			for (n = 0; n < sizeof(name_cols) / sizeof(name_cols[0]); n++) {
				if (try_bind_ids_from_query(cams, nr_cams, db,
							    tables[t], id_cols[i],
							    name_cols[n]))
					return;
			}
		}
	}
}

static void try_bind_ids(struct ss_camera *cams, unsigned int nr_cams)
{
	char dbs[MAX_DB_PATHS][SS_CAMERA_PATH_LEN];
	unsigned int nr_dbs;
	unsigned int i;

	find_system_dbs(dbs, &nr_dbs);
	for (i = 0; i < nr_dbs; i++)
		try_bind_ids_from_db(cams, nr_cams, dbs[i]);
}

int ss_camera_discover(struct ss_camera *cams, unsigned int max_cams,
		       unsigned int *nr_cams)
{
	if (!cams || !nr_cams)
		return -EINVAL;

	*nr_cams = 0;
	scan_volume_recordings(cams, max_cams, nr_cams);
	try_bind_ids(cams, *nr_cams);
	return 0;
}

int ss_camera_apply_group_map(struct ss_camera *cams, unsigned int nr_cams,
			      const char *path)
{
	char line[512];
	FILE *fp;

	if (!path || !*path)
		return 0;

	fp = fopen(path, "r");
	if (!fp)
		return -errno;

	while (fgets(line, sizeof(line), fp)) {
		char *p = trim(line);
		char *name;
		int group;
		unsigned int i;

		if (!*p || *p == '#')
			continue;
		group = strtol(p, &name, 0);
		name = trim(name);
		if (!*name)
			continue;

		for (i = 0; i < nr_cams; i++) {
			if (!strcmp(cams[i].name, name)) {
				cams[i].group = group;
				break;
			}
		}
	}

	fclose(fp);
	return 0;
}

int ss_camera_find(struct ss_camera *cams, unsigned int nr_cams,
		   int id, const char *name, int group)
{
	unsigned int i;

	for (i = 0; i < nr_cams; i++) {
		if (id >= 0 && cams[i].id == id)
			return (int)i;
		if (name && *name && !strcmp(cams[i].name, name))
			return (int)i;
		if (group >= 0 && cams[i].group == group)
			return (int)i;
	}

	return -1;
}

void ss_camera_print_json(const struct ss_camera *cams, unsigned int nr_cams)
{
	unsigned int i;

	printf("{\"cameras\":[");
	for (i = 0; i < nr_cams; i++) {
		printf("%s{\"id\":%d,\"group\":%d,\"name\":",
		       i ? "," : "", cams[i].id, cams[i].group);
		json_escape(cams[i].name);
		printf(",\"camera_dir\":");
		json_escape(cams[i].camera_dir);
		printf("}");
	}
	printf("]}\n");
}
