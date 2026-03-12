#define _GNU_SOURCE

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

    Matrix *mat = init_mat(arena, 7, 3, true);
    Matrix *mat2 = init_mat(arena, 7, 3, true);
    // Matrix *res = init_mat(arena, 7, 3, true);
    rng_seed((u64)time(NULL));
    mat_fill_rand(mat, sqrt(2.0 / mat->cols));
    mat_fill_rand(mat2, sqrt(2.0 / mat->cols));

    mat_print(mat, "a");
    mat_print(mat2, "b");

    mat_add(mat, mat, mat2);
    mat_print(mat, "a");

    arena_delete(arena);

    return 0;
}
