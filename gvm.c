#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <math.h>
#include <sys/socket.h>
#include <netdb.h>
#include <sys/types.h>
#include <assert.h>
#include <sys/un.h>
#include <stdint.h>
#include <sched.h>
#include <stdio.h>

#include "helper.h"
#include "ringbuffer.h"
#include "utils.h"

#define CUDA_EVENT_BATCH_SIZE 64
#define CUDA_EVENT_INTERVAL_MS 8

extern entry_t cuda_library_entry[];

// FIXME: this is not thread safe
static int64_t g_cuda_mem_allocated = 0;
static int64_t g_cuda_mem_total = 0UL;

static const size_t g_rb_size = 1048576;

static int g_uvmfd = -1;

static struct ringbuffer g_event_rb;

static _Atomic size_t submitted;
static _Atomic size_t submitted_at_event;
static _Atomic size_t timestamp_at_event;

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
		CUdriverProcAddressQueryResult *symbolStatus);

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
		CUdriverProcAddressQueryResult *symbolStatus);

CUresult cuMemAlloc(void **devPtr, size_t size) {
	CUresult ret = CUDA_SUCCESS;

	if (g_cuda_mem_total == 0) {
		size_t _cuda_mem_free = 0;
		size_t _cuda_mem_total = 0;
		CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetInfo_v2, &_cuda_mem_free, &_cuda_mem_total);
		g_cuda_mem_total = _cuda_mem_total;
	}
	if (g_uvmfd < 0) {
		CUdevice device;
		CUuuid uuid;
		ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
		if (ret != CUDA_SUCCESS)
			fprintf(stderr, "cuCtxGetDevice: error code %d\n", ret);
		ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceGetUuid, &uuid, device);
		if (ret != CUDA_SUCCESS)
			fprintf(stderr, "cuDeviceGetUuid: error code %d\n", ret);

		g_uvmfd = find_initialized_uvm(uuid);
		printf("Find uvmfd at %d\n", g_uvmfd);
	}

	if (g_cuda_mem_allocated + size > g_cuda_mem_total) {
		fprintf(stderr, "[INTERCEPTOR] cuMemAlloc: out of memory.\n");
		return CUDA_ERROR_OUT_OF_MEMORY;
	}

	ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, devPtr, size, CU_MEM_ATTACH_GLOBAL);
	if (ret != CUDA_SUCCESS) {
		fprintf(stderr, "[INTERCEPTOR] cuMemAllocManaged: out of memory.\n");
		return ret;
	}

	g_cuda_mem_allocated += size;
	printf("total cuda memory allocated: %lluMB\n", g_cuda_mem_allocated / 1024 / 1024);

	return ret;
}

CUresult cuMemAllocAsync(void **devPtr, size_t size, CUstream stream) {
	(void)stream; // suppress warning about unused stream

	CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, devPtr, size, CU_MEM_ATTACH_GLOBAL);
	if (ret != CUDA_SUCCESS) {
		fprintf(stderr, "[INTERCEPTOR] cuMemAllocAsync: out of memory.\n");
		return ret;
	}

	g_cuda_mem_allocated += size;
	printf("total cuda memory allocated: %lluMB\n", g_cuda_mem_allocated / 1024 / 1024);

	return ret;
}

CUresult cuMemFree(void *devPtr) {
	void *base;
	size_t size;

	if (CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetAddressRange_v2, &base, &size, devPtr) == CUDA_SUCCESS)
		g_cuda_mem_allocated -= size;

	return CUDA_ENTRY_CALL(cuda_library_entry, cuMemFree_v2, devPtr);
}

CUresult cuLaunchKernel(const void* f,
		unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
		unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
		unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
	CUresult ret;
	CUdevice device;
	CUuuid uuid;
 	struct ringbuffer_element *elem;
	size_t new_submitted_at_event;
	size_t old_submitted_at_event;
	size_t new_timestamp_at_event;
	size_t old_timestamp_at_event;

	ret = CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchKernel, f, gridDimX, gridDimY, gridDimZ,
			blockDimX, blockDimY, blockDimZ,
			sharedMemBytes, hStream, kernelParams, extra);

	new_submitted_at_event = atomic_fetch_add_explicit(&submitted, 1, memory_order_release) + 1;
	old_submitted_at_event = atomic_load_explicit(&submitted_at_event, memory_order_acquire);
	new_timestamp_at_event = gettime_ms();
	old_timestamp_at_event = atomic_load_explicit(&timestamp_at_event, memory_order_acquire);
	if (new_timestamp_at_event > old_timestamp_at_event + CUDA_EVENT_INTERVAL_MS &&
			atomic_compare_exchange_strong(&timestamp_at_event, &old_timestamp_at_event, new_timestamp_at_event)) {
		if (rb_enqueue_start(&g_event_rb, &elem, true) != 0)
 			fprintf(stderr, "rb_enqueue: Unknown error\n");

		if (rb_elem_is_valid(elem))
			fprintf(stderr, "rb_elem_is_valid: Unknown error\n");

		ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
		if (ret != CUDA_SUCCESS)
			fprintf(stderr, "cuCtxGetDevice: error code %d\n", ret);
		ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceGetUuid, &uuid, device);
		if (ret != CUDA_SUCCESS)
			fprintf(stderr, "cuDeviceGetUuid: error code %d\n", ret);

		elem->uuid = uuid;
		elem->submitted_during_event = new_submitted_at_event - old_submitted_at_event;

		if (g_uvmfd >= 0 &&
				update_event_count(g_uvmfd, uuid, UVM_SUBMIT_KERNEL_EVENT, UVM_ADD_EVENT_COUNT, elem->submitted_during_event) < 0) {
			fprintf(stderr, "update_event_count: unknown reason\n");
		}

		CUDA_ENTRY_CALL(cuda_library_entry, cuEventRecord, elem->event, hStream);

		if (rb_enqueue_end(&g_event_rb, elem) != 0)
			fprintf(stderr, "rb_enqueue_end: Unknown error\n");

		if (atomic_fetch_add_explicit(&submitted_at_event, elem->submitted_during_event, memory_order_release) != old_submitted_at_event)
			fprintf(stderr, "atomic_fetch_add_explicit: Unknown error\n");
	}

	return ret;
}

static entry_t hooked_cuda_entry[] = {
	{ .name = "cuMemAlloc", .fn_ptr = cuMemAlloc },
	{ .name = "cuMemAllocAsync", .fn_ptr = cuMemAllocAsync },
	{ .name = "cuMemFree", .fn_ptr = cuMemFree },
	{ .name = "cuLaunchKernel", .fn_ptr = cuLaunchKernel },
	{ .name = "cuGetProcAddress", .fn_ptr = cuGetProcAddress },
	{ .name = "cuGetProcAddress_v2", .fn_ptr = cuGetProcAddress_v2 },
};

static size_t hooked_cuda_entry_num = sizeof(hooked_cuda_entry) / sizeof(entry_t);

void *load_hooked_cuda_entry(const char *symbol) {
	size_t hooked_cuda_entry_index;
	void *fn_ptr = NULL;

	for (hooked_cuda_entry_index = 0; hooked_cuda_entry_index < hooked_cuda_entry_num; ++hooked_cuda_entry_index) {
		if (strcmp(symbol, hooked_cuda_entry[hooked_cuda_entry_index].name) == 0) {
			fn_ptr = hooked_cuda_entry[hooked_cuda_entry_index].fn_ptr;
			break;
		}
	}

	return fn_ptr;
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
		CUdriverProcAddressQueryResult *symbolStatus) {
	CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuGetProcAddress, symbol, pfn, cudaVersion, flags, symbolStatus);
	void *fn_ptr = NULL;
	if (ret == CUDA_SUCCESS) {
		fn_ptr = load_hooked_cuda_entry(symbol);
		if (fn_ptr) {
			*pfn = fn_ptr;
		}
		return CUDA_SUCCESS;
	}
	return ret;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
		CUdriverProcAddressQueryResult *symbolStatus) {
	CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuGetProcAddress_v2, symbol, pfn, cudaVersion, flags, symbolStatus);
	void *fn_ptr = NULL;
	if (ret == CUDA_SUCCESS) {
		fn_ptr = load_hooked_cuda_entry(symbol);
		if (fn_ptr) {
			*pfn = fn_ptr;
		}
		return CUDA_SUCCESS;
	}
	return ret;
}

static pthread_t event_thread;
static volatile bool running;

static void *event_handler(void *arg) {
	struct ringbuffer *rb = (struct ringbuffer *)arg;
	struct ringbuffer_element *elem = NULL;

	while (running) {
		while (rb_peek(rb, &elem, false) == 0) {
			if (!rb_elem_is_valid(elem)) {
				fprintf(stderr, "rb_elem_is_valid: Unknown error\n");
				break;
			}

			if (CUDA_ENTRY_CALL(cuda_library_entry, cuEventQuery, elem->event) != CUDA_SUCCESS)
				break;

			if (g_uvmfd >= 0 && update_event_count(g_uvmfd, elem->uuid, UVM_END_KERNEL_EVENT, UVM_ADD_EVENT_COUNT, elem->submitted_during_event) < 0) {
				fprintf(stderr, "update_event_count: unknown reason\n");
			}

			if (rb_dequeue(rb, elem) != 0)
				fprintf(stderr, "rb_dequeue: Unknown error\n");
		}
	}

	return NULL;
}

__attribute__((constructor))
void init(void) {
	atomic_store_explicit(&submitted, 0, memory_order_relaxed);
	running = true;
	rb_init(&g_event_rb, g_rb_size, "End");
	if (pthread_create(&event_thread, NULL, event_handler, &g_event_rb) != 0) {
		perror("pthread_create failed");
		exit(1);
	}
}

__attribute__((destructor))
void fini(void) {
	running = false;
	if (pthread_join(event_thread, NULL) != 0)
		perror("pthread_join failed");

	rb_deinit(&g_event_rb);
}
