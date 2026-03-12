#define _GNU_SOURCE
#include "./include/arena.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

Arena *arena_init(u64 reserve_size, u64 commit_size) {
    reserve_size = ALIGN_UP2(reserve_size, sysconf(_SC_PAGESIZE));
    commit_size = ALIGN_UP2(commit_size, sysconf(_SC_PAGESIZE));

    Arena *arena = (Arena *)mem_reserve(reserve_size);
    if (!arena) abort();
    if (!mem_commit(arena, commit_size)) abort();

    arena->reserve_size = reserve_size;
    arena->commit_size = commit_size;
    arena->commited = commit_size;
    arena->pos = ARENA_BASE_POS;

    return arena;
} /* arena_init() */

void *arena_push(Arena *arena, u64 size, b32 zero) {
    // pos_aligned is where the newly allocated memory will start
    u64 pos_aligned = ALIGN_UP2(arena->pos, ARENA_ALIGN);
    u64 new_pos = pos_aligned + size;

    if (new_pos > arena->reserve_size) abort();

    // check if we need to commit more mem
    if (new_pos > arena->commited) {
        // round new commit size up to page size
        u64 new_commited = new_pos;
        new_commited += arena->commit_size - 1;
        new_commited -= new_commited % arena->commit_size;
        new_commited = MIN(new_commited, arena->reserve_size);

        u8 *ptr = (u8 *)arena + arena->commited;
        u64 commit_size = new_commited - arena->commited;

        if (!mem_commit(ptr, commit_size)) abort();
        arena->commited = new_commited;
    }

    arena->pos = new_pos;
    u8 *res = (u8 *)arena + pos_aligned;

    // zero out if needed
    if (zero) {
        memset(res, 0, size);
    }
    return res;
} /* arena_push() */

void arena_pop(Arena *arena, u64 size) {
    size = MIN(size, arena->pos - ARENA_BASE_POS);
    arena->pos -= size;
} /* arena_pop() */

void arena_pop_to(Arena *arena, u64 pos) {
    // check pos passed in is before the current pos
    u64 size = pos < arena->pos ? arena->pos - pos : 0;
    arena_pop(arena, size);
} /* arena_pop_to() */

void arena_clear(Arena *arena) {
    arena->pos = ARENA_BASE_POS;
}
void arena_delete(Arena *arena) {
    mem_release(arena, arena->reserve_size);
}

ArenaTemp arena_temp_begin(Arena *arena) {
    return (ArenaTemp){.arena = arena, .start = arena->pos};
} /* arena_temp_begin() */

void arena_temp_end(ArenaTemp *temp) {
    arena_pop_to(temp->arena, temp->start);
} /* arena_temp_end() */

void *mem_reserve(u64 size) {
    // reserve virtual mem from OS, set it all to unusable at start
    void *pos = mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (pos == MAP_FAILED) {
        return NULL;
    }
    return pos;
} /* mem_reserve() */

b32 mem_commit(void *ptr, u64 size) {
    // commit = mark usable for arena
    return mprotect(ptr, size, PROT_READ | PROT_WRITE) == 0;
} /* mem_commit() */

b32 mem_decommit(void *ptr, u64 size) {
    // prevent accidentally touching decommited area
    return mprotect(ptr, size, PROT_NONE) == 0 && madvise(ptr, size, MADV_DONTNEED) == 0;
} /* mem_commit() */

b32 mem_release(void *ptr, u64 size) {
    // unreserve and give back to OS
    return munmap(ptr, size) == 0;
} /* mem_release() */
