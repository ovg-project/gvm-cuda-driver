#ifndef RINGBUFFER_H
#define RINGBUFFER_H

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdatomic.h>

#include "subset.h"

extern entry_t cuda_library_entry[];

struct ringbuffer_element {
	CUevent event;
	CUuuid uuid;
	size_t submitted_during_event;
	_Atomic bool valid;  // used as a "commit" flag
};

struct ringbuffer {
	struct ringbuffer_element *array;
	size_t size;				   // capacity (number of elements)
	char name[16];
	_Atomic size_t read_index;	 // total number of dequeued elements
	_Atomic size_t write_index;	// total number of enqueued reservations
};

// Helper: yield a bit in blocking waits
static inline void rb_pause(void)
{
	sched_yield();
}

int rb_init(struct ringbuffer *rb, size_t size, char *name)
{
	if (!rb || size == 0)
		return -1;

	rb->array = (struct ringbuffer_element *)
		calloc(size, sizeof(struct ringbuffer_element));
	if (!rb->array)
		return -2;

	rb->size = size;
	strncpy(rb->name, name, 16);
	atomic_store_explicit(&rb->read_index, 0, memory_order_relaxed);
	atomic_store_explicit(&rb->write_index, 0, memory_order_relaxed);

	// valid flags already zeroed by calloc
	return 0;
}

int rb_enqueue_start(struct ringbuffer *rb,
					 struct ringbuffer_element **out_elem,
					 bool blocking)
{
	if (!rb || !rb->array)
		return -1;

	for (;;) {
		size_t write = atomic_load_explicit(&rb->write_index,
											memory_order_relaxed);
		size_t read  = atomic_load_explicit(&rb->read_index,
											memory_order_acquire);

		// full if outstanding >= capacity
		if (write - read >= rb->size) {
			if (!blocking)
				return -1;  // full (non-blocking)
			rb_pause();
			continue;
		}

		size_t desired = write + 1;
		if (!atomic_compare_exchange_weak_explicit(
				&rb->write_index,
				&write,
				desired,
				memory_order_acq_rel,
				memory_order_relaxed)) {
			// lost race, retry
			continue;
		}

		// We now own index "write"
		size_t pos = write % rb->size;
		struct ringbuffer_element *elem = &rb->array[pos];

		// Wait until slot is truly free (previous consumer finished).
		while (atomic_load_explicit(&elem->valid, memory_order_acquire)) {
			rb_pause();
		}

		// Create events; leave valid=false so consumer won't see it yet.
		CUresult res;
		res = CUDA_ENTRY_CALL(cuda_library_entry, cuEventCreate, &elem->event, 0x0);
		if (res != CUDA_SUCCESS) {
			// If you want softer failure, return an error instead of abort.
			abort();
		}

		if (out_elem)
			*out_elem = elem;

		return 0;
	}
}

int rb_enqueue_end(struct ringbuffer *rb,
				   struct ringbuffer_element *elem)
{
	(void)rb; // not strictly needed, but kept for symmetry / future
	// Publish element as visible to consumer
	atomic_store_explicit(&elem->valid, true, memory_order_release);
	return 0;
}

int rb_peek(struct ringbuffer *rb,
			struct ringbuffer_element **out_elem,
			bool blocking)
{
	if (!rb || !rb->array || !out_elem)
		return -1;

	for (;;) {
		size_t read  = atomic_load_explicit(&rb->read_index,
											memory_order_relaxed);
		size_t write = atomic_load_explicit(&rb->write_index,
											memory_order_acquire);

		if (read == write) {
			// queue logically empty (no reserved slots)
			if (!blocking)
				return -1;  // empty
			rb_pause();
			continue;
		}

		size_t pos = read % rb->size;
		struct ringbuffer_element *elem = &rb->array[pos];

		bool v = atomic_load_explicit(&elem->valid, memory_order_acquire);
		if (!v) {
			// Slot reserved but not yet published (producer hasn't called end).
			if (!blocking)
				return -1;  // nothing ready yet
			rb_pause();
			continue;
		}

		// Element is ready; DO NOT modify read_index or destroy events here.
		*out_elem = elem;
		return 0;
	}
}

int rb_dequeue(struct ringbuffer *rb,
				   struct ringbuffer_element *elem)
{
	if (!rb || !rb->array || !elem)
		return -1;

	// We expect elem to correspond to the current read_index slot.
	size_t read = atomic_load_explicit(&rb->read_index,
									   memory_order_relaxed);
	size_t pos  = read % rb->size;
	struct ringbuffer_element *slot = &rb->array[pos];

	// Optional safety check in debug builds:
	// if (slot != elem) abort();

	// At this point, consumer has presumably already used the events
	// (cuEventElapsedTime, cuEventSynchronize, etc.).
	if (slot->event)
		CUDA_ENTRY_CALL(cuda_library_entry, cuEventDestroy, slot->event);

	slot->event = NULL;

	atomic_store_explicit(&slot->valid, false, memory_order_release);

	// Advance read index (only consumer writes this)
	atomic_store_explicit(&rb->read_index, read + 1, memory_order_release);

	return 0;
}

size_t rb_size(struct ringbuffer *rb) {
	size_t write = atomic_load_explicit(&rb->write_index, memory_order_acquire);
	size_t read  = atomic_load_explicit(&rb->read_index, memory_order_acquire);
	return write - read;
}

void rb_deinit(struct ringbuffer *rb)
{
	if (!rb || !rb->array)
		return;

	// Clean up any remaining valid events
	for (size_t i = 0; i < rb->size; ++i) {
		struct ringbuffer_element *elem = &rb->array[i];

		bool v = atomic_load_explicit(&elem->valid, memory_order_acquire);
		if (!v)
			continue;

		// Wait for events to complete; user wanted deinit to "wait for all valid
		// events". We wait on end; adjust if you also want start.
		if (elem->event)
			CUDA_ENTRY_CALL(cuda_library_entry, cuEventSynchronize, elem->event);

		if (elem->event)
			CUDA_ENTRY_CALL(cuda_library_entry, cuEventDestroy, elem->event);

		elem->event = NULL;
		atomic_store_explicit(&elem->valid, false, memory_order_relaxed);
	}

	free(rb->array);
	rb->array = NULL;
	rb->size = 0;
	atomic_store_explicit(&rb->read_index, 0, memory_order_relaxed);
	atomic_store_explicit(&rb->write_index, 0, memory_order_relaxed);
}

bool rb_elem_is_valid(const struct ringbuffer_element *elem) {
	return atomic_load_explicit(&elem->valid, memory_order_acquire);
}

#endif
