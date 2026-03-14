#include "./include/numc.h"
#include "./include/prng.h"

// allocation & init
Matrix *mat_init(Arena *arena, const u64 rows, const u64 cols, const bool zeroed) {
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
        fprintf(stderr, "Matrix Addition Error: Incompatible Sizes");
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
        fprintf(stderr, "Matrix Subtraction Error: Incompatible Sizes\n");
        return;
    }

    for (u64 row = 0; row < a->rows; row++) {
        for (u64 col = 0; col < a->cols; col++) {
            *MAT_AT(dest, row, col) = *MAT_AT(a, row, col) - *MAT_AT(b, row, col);
        }
    }
}

void mat_mul(Matrix *dest, const Matrix *a, const Matrix *b) {
    for (u64 row = 0; row < dest->rows; row++) {
        for (u64 i = 0; i < a->cols; i++) { // take advantage of caching
            f64 a_val = *MAT_AT(a, row, i);
            for (u64 col = 0; col < dest->cols; col++) {
                *MAT_AT(dest, row, col) += a_val * (*MAT_AT(b, i, col));
            }
        }
    }
}

void mat_mul_transpose(Matrix *dest, Matrix *a, Matrix *b, b32 a_transp, b32 b_transp) {
    u64 a_rows = a_transp ? a->cols : a->rows;
    u64 a_cols = a_transp ? a->rows : a->cols;
    // u64 b_rows = b_transp ? b->cols : b->rows;
    u64 b_cols = b_transp ? b->rows : b->cols;

    mat_fill(dest, 0);

    for (u64 row = 0; row < a_rows; row++) {
        for (u64 col = 0; col < b_cols; col++) {
            for (u64 i = 0; i < a_cols; i++) {
                f64 a_val = a_transp ? (*MAT_AT(a, i, row)) : (*MAT_AT(a, row, i));
                f64 b_val = b_transp ? (*MAT_AT(b, col, i)) : (*MAT_AT(b, i, col));
                *MAT_AT(dest, row, col) += a_val * b_val;
            }
        }
    }
}

void mat_scale(Matrix *dest, const Matrix *a, const f64 scalar) {
    for (u64 row = 0; row < a->rows; row++) {
        for (u64 col = 0; col < a->cols; col++) {
            *MAT_AT(dest, row, col) *= scalar;
        }
    }
}

void mat_add_vec(Matrix *dest, const Matrix *a, const Matrix *bias) {
    if (a->cols != bias->cols || dest->cols != a->cols || dest->rows != a->rows) {
        fprintf(stderr, "Matrix Add Vector Error: Incompatible Sizes\n");
    }

    for (u64 row = 0; row < dest->rows; row++) {
        for (u64 col = 0; col < dest->cols; col++) {
            *MAT_AT(dest, row, col) += *MAT_AT(bias, 0, col);
        }
    }
}

// shape operations
Matrix *mat_transpose(Arena *arena, const Matrix *a) {
    if (!a) {
        fprintf(stderr, "Matrix Transpose Error: NULL Matrix Passed In");
        return NULL;
    }

    Matrix *new_mat = mat_init(arena, a->cols, a->rows, true);
    if (!new_mat) return NULL;

    for (u64 row = 0; row < new_mat->rows; row++) {
        for (u64 col = 0; col < new_mat->cols; col++) {
            *MAT_AT(new_mat, row, col) = *MAT_AT(a, col, row);
        }
    }
    return new_mat;
}

Matrix *mat_copy(Arena *arena, const Matrix *a) {
    Matrix *copy = mat_init(arena, a->rows, a->cols, true);
    if (!copy) return NULL;

    copy->rows = a->rows;
    copy->cols = a->cols;
    memcpy(copy->data, a->data, a->rows * a->cols * sizeof(*a->data));

    return copy;
}

Matrix mat_row(const Matrix *a, u64 const row) {
    return (Matrix){.rows = 1, .cols = a->cols, .data = &a->data[row * a->cols]};
}

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
