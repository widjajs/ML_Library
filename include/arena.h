#ifndef ARENA_H
#define ARENA_H

#include "utility.h"

#define KiB(n) ((u64)(n) << 10)
#define MiB(n) ((u64)(n) << 20)
#define GiB(n) ((u64)(n) << 30)

#define ARENA_BASE_POS (sizeof(Arena))
#define ARENA_ALIGN (sizeof(void *))

typedef struct {
    u64 reserve_size;
    u64 commit_size;
    u64 commited;
    u64 pos;
} Arena;

typedef struct {
    Arena *arena;
    u64 start;
} ArenaTemp;

Arena *arena_init(u64 reserve_size, u64 commit_size);
void *arena_push(Arena *arena, u64 size, b32 zero);
void arena_pop(Arena *arena, u64 size);
void arena_pop_to(Arena *arena, u64 pos);
void arena_clear(Arena *arena);
void arena_delete(Arena *arena);

ArenaTemp arena_temp_begin(Arena *arena);
void arena_temp_end(ArenaTemp *temp);

void *mem_reserve(u64 size);
b32 mem_commit(void *ptr, u64 size);
b32 mem_decommit(void *ptr, u64 size);
b32 mem_release(void *ptr, u64 size);

#endif
