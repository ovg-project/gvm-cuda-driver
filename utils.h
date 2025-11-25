#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <limits.h>  // PATH_MAX

#define UVM_IS_INITIALIZED 80
typedef struct
{
	bool					initialized;    // IN/OUT
	int						rmStatus;		// OUT
} UVM_IS_INITIALIZED_PARAMS;

typedef enum
{
	UVM_SUBMIT_KERNEL_EVENT = 0,
	UVM_END_KERNEL_EVENT,

	UVM_EVENT_TYPE_COUNT
} UVM_EVENT_TYPE;

typedef enum {
	UVM_ADD_EVENT_COUNT = 0,
	UVM_SET_EVENT_COUNT,

	UVM_UPDATE_EVENT_COUNT_TYPE_COUNT
} UVM_UPDATE_EVENT_COUNT_TYPE;

#define UVM_UPDATE_EVENT_COUNT 84
typedef struct
{
	CUuuid uuid;
	UVM_EVENT_TYPE type;
	UVM_UPDATE_EVENT_COUNT_TYPE op;
	uint64_t value;

	int rmStatus;
} UVM_UPDATE_EVENT_COUNT_PARAMS;

int find_initialized_uvm(void)
{
	pid_t pid = getpid();
	char fd_dir[64];
	const char *target_path = "/dev/nvidia-uvm";
	int ret = -1;

	snprintf(fd_dir, sizeof(fd_dir), "/proc/%d/fd", (int)pid);

	DIR *dir = opendir(fd_dir);
	if (!dir) {
		fprintf(stderr, "Failed to open %s: %s\n", fd_dir, strerror(errno));
		return ret;
	}

	struct dirent *entry;
	while ((entry = readdir(dir)) != NULL) {
		char link_path[PATH_MAX];
		char fd_path[PATH_MAX];
		ssize_t len;
		int fd;

		// Skip "." and ".."
		if (entry->d_name[0] == '.' &&
			(entry->d_name[1] == '\0' ||
			 (entry->d_name[1] == '.' && entry->d_name[2] == '\0'))) {
			continue;
		}

		// /proc/<pid>/fd/<fd>
		snprintf(link_path, sizeof(link_path), "%s/%s", fd_dir, entry->d_name);

		len = readlink(link_path, fd_path, sizeof(fd_path) - 1);
		if (len < 0) {
			// could log if you care; original code just continued on error_code
			continue;
		}
		fd_path[len] = '\0';

		if (strcmp(fd_path, target_path) != 0)
			continue;

		fd = atoi(entry->d_name);
		if (fd < 0) {
			fprintf(stderr, "Invalid file descriptor: %d\n", fd);
			continue;
		}

		UVM_IS_INITIALIZED_PARAMS params;
		memset(&params, 0, sizeof(params));
		// If initialized is a bool/NvBool, zero-init is enough; otherwise:
		// params.initialized = 0;
		// params.rmStatus	= 0;

		if (ioctl(fd, UVM_IS_INITIALIZED, &params) == 0) {
			if (params.initialized) {
				ret = fd;
			}
		} else {
			fprintf(stderr, "Failed to check UVM initialization on fd %d: %s\n",
					fd, strerror(errno));
		}
	}

	closedir(dir);
	return ret;
}

int update_event_count(int uvmfd, CUuuid uuid, UVM_EVENT_TYPE type, UVM_UPDATE_EVENT_COUNT_TYPE op, uint64_t value) {
	int ret = 0;

	UVM_UPDATE_EVENT_COUNT_PARAMS params = {
		.uuid = uuid,
		.type = type,
		.op = op,
		.value = value,
		.rmStatus = 0
	};

	if ((ret = ioctl(uvmfd, UVM_UPDATE_EVENT_COUNT, &params)) != 0)
		return ret;

	if (params.rmStatus != 0)
		fprintf(stderr, "uvm_ioctl: error code %d\n", params.rmStatus);

	return 0;
}

size_t gettime_ms(void) {
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	return (size_t)(ts.tv_sec) * 1000 + (ts.tv_nsec) / 1000000;
}

#endif
