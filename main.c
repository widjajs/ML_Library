#define _GNU_SOURCE

#include <assert.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "./include/arena.h"
#include "./include/candas.h"
#include "./include/numc.h"
#include "./include/prng.h"

int main() {
    // u32 img_magic, img_count, img_rows, img_cols;

    // FILE *fp_images = fopen("./dataset/train-images.idx3-ubyte", "rb");
    Arena *arena = arena_init(GiB(1), sysconf(_SC_PAGESIZE));

    // read_image_headers(fp_images, &img_magic, &img_count, &img_rows, &img_cols);
    // printf("%d, %d, %d, %d\n", img_magic, img_count, img_rows, img_cols);

    // u64 total_pixels = img_count * img_rows * img_cols;
    // u8 *training_images = arena_push(arena, total_pixels, false);

    // read_images(fp_images, training_images, total_pixels);
    // draw_mnist_ascii(training_images + ((img_count - 215) * 28 * 28));
    // draw_mnist_color(training_images + ((img_count - 215) * 28 * 28));

    // u32 label_magic, labels_size;
    // FILE *fp_labels = fopen("./dataset/train-labels.idx1-ubyte", "rb");
    // read_label_headers(fp_labels, &label_magic, &labels_size);

    // printf("%d %d\n", label_magic, labels_size);

    // u8 *training_labels = arena_push(arena, labels_size, false);

    // read_labels(fp_labels, training_labels, labels_size);
    // printf("%d\n", training_labels[img_count - 40323]);

    // fclose(fp_images);
    // fclose(fp_labels);
    Matrix *a4 = mat_init(arena, 3, 2, true);
    Matrix *b4 = mat_init(arena, 2, 3, true);
    Matrix *dest4 = mat_init(arena, 2, 2, true);

    *MAT_AT(a4, 0, 0) = 1;
    *MAT_AT(a4, 0, 1) = 4;
    *MAT_AT(a4, 1, 0) = 2;
    *MAT_AT(a4, 1, 1) = 5;
    *MAT_AT(a4, 2, 0) = 3;
    *MAT_AT(a4, 2, 1) = 6;

    *MAT_AT(b4, 0, 0) = 1;
    *MAT_AT(b4, 0, 1) = 2;
    *MAT_AT(b4, 0, 2) = 3;
    *MAT_AT(b4, 1, 0) = 4;
    *MAT_AT(b4, 1, 1) = 5;
    *MAT_AT(b4, 1, 2) = 6;

    // A^T * B^T = (B*A)^T
    // B*A = [[1,2,3],[4,5,6]] * [[1,4],[2,5],[3,6]] = [[14,32],[32,77]]
    // (B*A)^T = [[14,32],[32,77]] (symmetric in this case)
    mat_mul_transpose(dest4, a4, b4, true, true);
    mat_print(dest4, "Test4: A^T * B^T (expect [[14,32],[32,77]])");

    Matrix *c = mat_copy(arena, dest4);
    mat_print(c, "copied");

    Matrix view = mat_row(c, 0);
    mat_print(&view, "view");

    arena_delete(arena);

    return 0;
}
