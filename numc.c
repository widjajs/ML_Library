#include "./include/numc.h"
#include "./include/prng.h"
#include "include/arena.h"
#include "include/utility.h"

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

    memcpy(copy->data, a->data, a->rows * a->cols * sizeof(*a->data));

    return copy;
}

Matrix mat_row(const Matrix *a, u64 const row) {
    return (Matrix){.rows = 1, .cols = a->cols, .data = &a->data[row * a->cols]};
}

// activation functions
void mat_relu(Matrix *dest, const Matrix *a) {
    for (u64 row = 0; row < a->rows; row++) {
        for (u64 col = 0; col < a->cols; col++) {
            *MAT_AT(dest, row, col) = MAX(0, *MAT_AT(a, row, col));
        }
    }
}

void mat_relu_backward(Matrix *dest, const Matrix *grad, const Matrix *forward) {
    for (u64 row = 0; row < grad->rows; row++) {
        for (u64 col = 0; col < grad->cols; col++) {
            *MAT_AT(dest, row, col) =
                *MAT_AT(grad, row, col) * (*MAT_AT(forward, row, col) > 0);
        }
    }
}

void mat_softmax(Matrix *dest, const Matrix *a) {
    // find max
    f64 max = *MAT_AT(a, 0, 0);
    for (u64 row = 0; row < a->rows; row++) {
        for (u64 col = 0; col < a->cols; col++) {
            f64 val = *MAT_AT(a, row, col);
            max = MAX(max, val);
        }
    }

    // exponentiate everything and subtract max to avoid overflow
    f64 sum = 0.0;
    for (u64 row = 0; row < a->rows; row++) {
        for (u64 col = 0; col < a->cols; col++) {
            f64 exponentiated = exp(*MAT_AT(a, row, col) - max);
            *MAT_AT(dest, row, col) = exponentiated;
            sum += exponentiated;
        }
    }

    // normalize
    for (u64 row = 0; row < a->rows; row++) {
        for (u64 col = 0; col < a->cols; col++) {
            *MAT_AT(dest, row, col) /= sum;
        }
    }
}

Matrix *mat_one_hot(Arena *arena, const u8 *labels, const u64 n, const u64 num_classes) {
    Matrix *one_hot = mat_init(arena, n, num_classes, 0); // n x num_classes
    for (u64 i = 0; i < n; i++) {
        *MAT_AT(one_hot, i, labels[i]) = 1;
    }
    return one_hot;
}

// reduction/loss
f64 mat_cross_entropy(Arena *arena, const Matrix *probs, const u8 *labels) {
    // not used in training pipeline; just a good inidicator for printing progress
    ArenaTemp temp = arena_temp_begin(arena);
    Matrix *one_hot = mat_one_hot(temp.arena, labels, probs->rows, probs->cols);
    f64 res = 0.0;
    for (u64 row = 0; row < probs->rows; row++) {
        for (u64 col = 0; col < probs->cols; col++) {
            res += *MAT_AT(one_hot, row, col) * log(*MAT_AT(probs, row, col));
        }
    }
    arena_temp_end(&temp);
    return -res / probs->rows;
}

void mat_softmax_grad(Arena *arena, Matrix *dest, const Matrix *probs, const u8 *labels) {
    ArenaTemp temp = arena_temp_begin(arena);
    Matrix *one_hot = mat_one_hot(temp.arena, labels, probs->rows, probs->cols);
    for (u64 row = 0; row < probs->rows; row++) {
        for (u64 col = 0; col < probs->cols; col++) {
            // predicted (probs) - actual (one_hot labels)
            *MAT_AT(dest, row, col) =
                (*MAT_AT(probs, row, col) - *MAT_AT(one_hot, row, col)) / probs->rows;
        }
    }
    arena_temp_end(&temp);
}

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
