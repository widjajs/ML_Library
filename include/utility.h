#ifndef UTILITY
#define UTILITY

#include <stdint.h>

// signed ints
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

// unsigned ints
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

// bools
typedef i8 b8;
typedef i32 b32;

// floating point numbers
typedef float f32;
typedef double f64;

// some useful macros
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define ALIGN_UP2(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))
#define ALIGN_DOWN2(n, p) (((u64)(n)) & (~((u64)(p) - 1)))
#define M_PI 3.14159265358979323846

#endif
