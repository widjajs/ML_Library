#ifndef NUMC
#define NUMC

#include "arena.h"

typedef struct {
    u64 rows;
    u64 cols;
    f64 *data;
} Matrix;

// indexing
#define MAT_AT(mat, row, col) ((mat).data + ((row) * (mat).cols + (col)))

// allocation & init
Matrix init_mat(Arena *arena, const u64 rows, const u64 cols, const bool zeroed);
void mat_fill(Matrix a, const f64 bval);
void mat_fill_rand(Matrix a, const f64 scale);

// matrix arithmetic operations
void mat_add(Matrix dest, const Matrix a, const Matrix b);
void mat_sub(Matrix dest, const Matrix a, const Matrix b);
void mat_mul(Matrix dest, const Matrix a, const Matrix b);
void mat_scale(Matrix dest, const Matrix a, const f64 scalar);
void mat_add_vec(Matrix dest, const Matrix a, const Matrix bias);

// shape operations
Matrix mat_transpose(const Matrix a);
Matrix mat_copy(Arena *arena, const Matrix a);
Matrix mat_row(const Matrix a, u64 const row);

// activation functions
void mat_relu(Matrix dest, const Matrix a);
void mat_relu_backward(Matrix dest, const Matrix gradient, const Matrix forward);
void mat_softmax(Matrix dest, const Matrix a);

// reduction/loss
f64 mat_cross_entropy(const Matrix probs, const u8 *labels, const u64 n);
void mat_softmax_grad(Matrix dest, const Matrix probs, const u8 *labels, const u64 n);

// debug
void mat_print(Matrix a, const char *name);
void mat_shape(Matrix a, const char *name);

#endif
