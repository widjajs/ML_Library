#include "./include/numc.h"
#include "./include/prng.h"

// allocation & init
Matrix *init_mat(Arena *arena, const u64 rows, const u64 cols, const bool zeroed) {
    if (!arena) return NULL;

    Matrix *mat = arena_push(arena, sizeof(Matrix), false);
    if (!mat) return NULL;

    f64 *data = arena_push(arena, rows * cols * sizeof(u64), zeroed);
    if (!data) return NULL;

    mat->rows = rows;
    mat->cols = cols;
    mat->data = data;

    return mat;
}

void mat_fill(Matrix *a, const f64 val) {
    for (u64 row = 0; row < a->rows; row++) {
        for (u64 col = 0; col < a->cols; col++) {
            *MAT_AT(a, row, col) = val;
        }
    }
}

void mat_fill_rand(Matrix *a, const f64 scale) {
    for (u64 row = 0; row < a->rows; row++) {
        for (u64 col = 0; col < a->cols; col++) {
            *MAT_AT(a, row, col) = randn() * scale;
        }
    }
}

// matrix arithmetic operations
void mat_add(Matrix *dest, const Matrix *a, const Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        perror("Matrix Addition Error: incompatible sizes");
        return;
    }

    for (u64 row = 0; row < a->rows; row++) {
        for (u64 col = 0; col < a->cols; col++) {
            *MAT_AT(dest, row, col) = *MAT_AT(a, row, col) + *MAT_AT(b, row, col);
        }
    }
}
void mat_sub(Matrix *dest, const Matrix *a, const Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        perror("Matrix Subtraction Error: incompatible sizes");
        return;
    }

    for (u64 row = 0; row < a->rows; row++) {
        for (u64 col = 0; col < a->cols; col++) {
            *MAT_AT(dest, row, col) = *MAT_AT(a, row, col) - *MAT_AT(b, row, col);
        }
    }
}
void mat_mul(Matrix *dest, const Matrix *a, const Matrix *b);
void mat_scale(Matrix *dest, const Matrix *a, const f64 scalar) {
    for (u64 row = 0; row < a->rows; row++) {
        for (u64 col = 0; col < a->cols; col++) {
            *MAT_AT(dest, row, col) *= scalar;
        }
    }
}
void mat_add_vec(Matrix *dest, const Matrix *a, const Matrix *bias);

// shape operations
Matrix mat_transpose(const Matrix *a);
Matrix mat_copy(Arena *arena, const Matrix *a);
Matrix mat_row(const Matrix *a, u64 const row);

// activation functions
void mat_relu(Matrix *dest, const Matrix *a);
void mat_relu_backward(Matrix *dest, const Matrix *gradient, const Matrix *forward);
void mat_softmax(Matrix *dest, const Matrix *a);

// reduction/loss
f64 mat_cross_entropy(const Matrix *probs, const u8 *labels, const u64 n);
void mat_softmax_grad(Matrix *dest, const Matrix *probs, const u8 *labels, const u64 n);

// debug
void mat_print(Matrix *a, const char *name) {
    if (!a || !a->data) return;

    printf("Printing: %s\n", name);
    u64 edge_elems = 3; // only print full matrix for up to 6 x 6
    for (u64 row = 0; row < a->rows; row++) {
        if (a->rows > edge_elems * 2 && row == edge_elems) {
            printf(" ...\n"); // if matrix too big just print "..." for middle rows
            row = a->rows - edge_elems - 1;
            continue;
        }

        printf(" [");
        for (u64 col = 0; col < a->cols; col++) {
            if (a->cols > edge_elems * 2 && col == edge_elems) {
                printf(" ...  ,"); // if matrix too big just print "..." for midddle cols
                col = a->cols - edge_elems - 1;
                continue;
            }

            printf("%9.5f", *MAT_AT(a, row, col));

            if (col < a->cols - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    }
}

void mat_shape(Matrix *a) {
    if (!a) return;
    printf("(%lu, %lu)\n", a->rows, a->cols);
}
