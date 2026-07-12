#ifndef SS_CAMERA_H
#define SS_CAMERA_H

#include <stddef.h>

#define SS_CAMERA_NAME_LEN 256
#define SS_CAMERA_PATH_LEN 512

struct ss_camera {
	int id;
	int group;
	char name[SS_CAMERA_NAME_LEN];
	char camera_dir[SS_CAMERA_PATH_LEN];
};

int ss_camera_discover(struct ss_camera *cams, unsigned int max_cams,
		       unsigned int *nr_cams);
int ss_camera_apply_group_map(struct ss_camera *cams, unsigned int nr_cams,
			      const char *path);
int ss_camera_find(struct ss_camera *cams, unsigned int nr_cams,
		   int id, const char *name, int group);
void ss_camera_print_json(const struct ss_camera *cams, unsigned int nr_cams);

#endif
