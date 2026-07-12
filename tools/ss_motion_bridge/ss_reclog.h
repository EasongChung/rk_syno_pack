#ifndef SS_RECLOG_H
#define SS_RECLOG_H

#include <stdbool.h>
#include <time.h>

struct ss_reclog_result {
	char path[512];
	unsigned int exists;
	unsigned int touched;
	unsigned int changed;
	int err;
};

int ss_reclog_mark_range(const char *camera_dir, time_t start, time_t stop,
			 bool backup, bool dry_run,
			 struct ss_reclog_result *results,
			 unsigned int max_results, unsigned int *nr_results);
int ss_reclog_mark_range_create_missing(const char *camera_dir,
					 time_t start, time_t stop,
					 bool backup, bool dry_run,
					 struct ss_reclog_result *results,
					 unsigned int max_results,
					 unsigned int *nr_results);

#endif
